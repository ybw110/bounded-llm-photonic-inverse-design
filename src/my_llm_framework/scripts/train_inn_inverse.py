from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_llm_framework.config import load_config, resolve_project_path
from my_llm_framework.data.processed import load_processed_split
from my_llm_framework.models.inn import INNInverseModel
from my_llm_framework.scripts.train_pnn_forward import spectra_to_target

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("训练 INN inverse model 需要安装 PyTorch：pip install torch") from exc


def set_seed(seed: int) -> None:
    """固定随机种子，尽量保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    split_data: dict[str, Any],
    active_dim: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """把 processed split 转成 INN 训练使用的 DataLoader。

    INN 的监督学习形式是：

    `complex spectrum -> normalized active parameters`
    """
    x = spectra_to_target(split_data["spectra"])
    y = split_data["params"][:, :active_dim].astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> float:
    """计算一个 split 上的平均参数 MSE。"""
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += float(loss.item()) * len(x)
            total_count += len(x)
    return total_loss / max(total_count, 1)


def train_one_domain(
    cfg: dict[str, Any],
    domain: str,
    processed_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """训练一个 domain 的 INN inverse model。"""
    domain_cfg = cfg["domains"][domain]
    train_cfg = cfg.get("inn_training", {})

    active_dim = int(domain_cfg["active_param_dim"])
    train_split = load_processed_split(processed_root, domain, "train")
    val_split = load_processed_split(processed_root, domain, "val")
    test_split = load_processed_split(processed_root, domain, "test")
    input_dim = spectra_to_target(train_split["spectra"][:1]).shape[1]

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 512))
    epochs = int(args.epochs or train_cfg.get("epochs", 80))
    learning_rate = float(args.learning_rate or train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-6))
    hidden_dim = int(args.hidden_dim or train_cfg.get("hidden_dim", 256))
    num_blocks = int(args.num_blocks or train_cfg.get("num_blocks", 4))
    dropout = float(args.dropout if args.dropout is not None else train_cfg.get("dropout", 0.05))
    patience = int(args.patience or train_cfg.get("patience", 12))
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 0))

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader = make_loader(train_split, active_dim, batch_size, True, num_workers)
    val_loader = make_loader(val_split, active_dim, batch_size, False, num_workers)
    test_loader = make_loader(test_split, active_dim, batch_size, False, num_workers)

    model = INNInverseModel(
        input_dim=input_dim,
        output_dim=active_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(4, patience // 3),
    )

    domain_output = output_root / domain
    domain_output.mkdir(parents=True, exist_ok=True)
    best_path = domain_output / "model_best.pt"
    last_path = domain_output / "model_last.pt"
    history_path = domain_output / "training_history.csv"

    best_val = float("inf")
    best_epoch = 0
    best_test = float("nan")
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        test_loss = evaluate(model, test_loader, device, loss_fn)
        scheduler.step(val_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_param_mse": train_loss,
            "val_param_mse": val_loss,
            "test_param_mse": test_loss,
            "learning_rate": current_lr,
        }
        history.append(row)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_test = test_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "domain": domain,
                    "input_dim": input_dim,
                    "active_dim": active_dim,
                    "output_dim": active_dim,
                    "hidden_dim": hidden_dim,
                    "num_blocks": num_blocks,
                    "dropout": dropout,
                    "param_names": domain_cfg["param_names"][:active_dim],
                    "summary": {
                        "best_epoch": best_epoch,
                        "best_val_param_mse": best_val,
                        "best_test_param_mse_at_val_best": best_test,
                        "train_rows": len(train_split["params"]),
                        "val_rows": len(val_split["params"]),
                        "test_rows": len(test_split["params"]),
                    },
                },
                best_path,
            )
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % args.print_every == 0 or improved:
            print(
                f"[{domain}] epoch={epoch:04d} "
                f"train_param_mse={train_loss:.6e} val_param_mse={val_loss:.6e} "
                f"test_param_mse={test_loss:.6e} best_val={best_val:.6e} lr={current_lr:.2e}"
            )

        if stale_epochs >= patience:
            print(f"[{domain}] early stopping at epoch {epoch}, best epoch {best_epoch}")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "domain": domain,
            "input_dim": input_dim,
            "active_dim": active_dim,
            "output_dim": active_dim,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "dropout": dropout,
            "param_names": domain_cfg["param_names"][:active_dim],
            "summary": {
                "best_epoch": best_epoch,
                "best_val_param_mse": best_val,
                "best_test_param_mse_at_val_best": best_test,
                "train_rows": len(train_split["params"]),
                "val_rows": len(val_split["params"]),
                "test_rows": len(test_split["params"]),
            },
        },
        last_path,
    )

    pd.DataFrame(history).to_csv(history_path, index=False)

    summary = {
        "domain": domain,
        "best_epoch": best_epoch,
        "best_val_param_mse": best_val,
        "best_test_param_mse_at_val_best": best_test,
        "last_epoch": history[-1]["epoch"],
        "train_rows": len(train_split["params"]),
        "val_rows": len(val_split["params"]),
        "test_rows": len(test_split["params"]),
        "input_dim": input_dim,
        "active_dim": active_dim,
        "best_model_path": str(best_path),
        "last_model_path": str(last_path),
        "history_path": str(history_path),
    }
    (domain_output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON，例如 configs/experiment_v1.json")
    parser.add_argument("--processed-root", default=None, help="processed dataset 目录；默认使用配置中的 processed_root")
    parser.add_argument("--output-root", default=None, help="模型输出目录；默认使用 inn_training.output_root")
    parser.add_argument("--domains", nargs="+", default=None, help="要训练的 domain；默认训练全部 domain")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="例如 cuda、cuda:0 或 cpu；默认自动选择")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("inn_training", {})
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 48))
    set_seed(seed)

    processed_root = Path(args.processed_root) if args.processed_root else Path(cfg["_processed_root_abs"])
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = resolve_project_path(cfg, train_cfg.get("output_root", "outputs/phase2_models/inn_inverse"))
    output_root.mkdir(parents=True, exist_ok=True)

    domains = args.domains or list(cfg["domains"].keys())
    summaries = []
    print(f"Training INN domains: {domains}")
    print(f"Processed root: {processed_root.resolve()}")
    print(f"Output root: {output_root.resolve()}")

    for domain in domains:
        summaries.append(train_one_domain(cfg, domain, processed_root, output_root, args))

    summary_path = output_root / "training_summary_all.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved INN training summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
