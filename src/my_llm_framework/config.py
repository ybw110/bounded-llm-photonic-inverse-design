from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """读取实验配置文件，并补充几个绝对路径字段。

    这个项目的配置文件放在 `configs/experiment_v1.json`。
    配置里很多路径是相对路径，例如旧项目已经迁入新项目后写成：

    `legacy/my_transefer_learning_framework`

    如果直接在不同工作目录运行脚本，相对路径很容易解析错。
    所以这里统一把几个常用路径都转换为绝对路径，后面的代码只需要
    使用这些补充字段即可。

    返回的 cfg 中新增字段含义：

    - `_config_path`：当前 JSON 配置文件的绝对路径
    - `_config_dir`：配置文件所在目录，也就是 `configs`
    - `_project_root`：新项目根目录，也就是 `MY-LLM-Framework`
    - `_legacy_root_abs`：旧项目 `legacy/my_transefer_learning_framework` 的绝对路径
    - `_processed_root_abs`：新项目生成的 processed dataset 目录
    """
    config_path = Path(path).resolve()

    # 用 UTF-8 读取，避免中文路径、中文注释或中文字段在 Windows 下乱码。
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 配置文件在 configs/ 下，所以它的父目录的父目录就是项目根目录。
    project_root = config_path.parent.parent

    # 把路径信息写回 cfg，方便后续模块复用。
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    cfg["_project_root"] = str(project_root)
    cfg["_legacy_root_abs"] = str((project_root / cfg["legacy_root"]).resolve())
    cfg["_processed_root_abs"] = str((project_root / cfg.get("processed_root", "data/processed/v1")).resolve())
    return cfg


def resolve_legacy_path(cfg: dict[str, Any], relative_path: str | Path) -> Path:
    """把旧项目中的相对路径转换为绝对路径。

    例如配置中写：

    `cylinder.mat`

    这个函数会把它转换成：

    `D:/论文写作/超表面+llm/MY-LLM-Framework/legacy/my_transefer_learning_framework/cylinder.mat`
    """
    return Path(cfg["_legacy_root_abs"]) / relative_path


def resolve_project_path(cfg: dict[str, Any], relative_path: str | Path) -> Path:
    """把项目内的相对路径转换为绝对路径。"""
    return Path(cfg["_project_root"]) / relative_path


def domain_frequency_grid(domain_cfg: dict[str, Any], stride: int) -> Any:
    """根据某个 domain 的频率范围生成下采样后的频率网格。

    旧框架的原始谱线通常是 301 个频点。
    V1 沿用旧框架的 `response_stride=10`，因此频率点会变成：

    0, 10, 20, ..., 300

    总共 31 个频点。这个函数保证频率数组和下采样后的复谱长度一致。
    """
    import numpy as np

    raw_points = int(domain_cfg["raw_points"])
    full_grid = np.linspace(
        float(domain_cfg["frequency_min_thz"]),
        float(domain_cfg["frequency_max_thz"]),
        raw_points,
    )
    return full_grid[::stride]
