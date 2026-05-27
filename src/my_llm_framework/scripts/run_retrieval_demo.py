from __future__ import annotations

import argparse
import json

import numpy as np

from my_llm_framework.config import domain_frequency_grid, load_config
from my_llm_framework.retrieval.domain_aware import DomainAwareRetriever, RetrievalWeights


def main() -> None:
    """运行一次 retrieval demo。

    这个脚本的用途是快速检查三件事：

    1. memory bank 是否能正确读取
    2. query 谱线是否能构造 physics summary
    3. domain-aware retrieval 是否能返回候选样本

    它不是正式 benchmark，只是单条 query 的人工检查入口。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    parser.add_argument("--memory", "--memory-bank", dest="memory", required=True, help="build_memory_bank 生成的 npz 文件")
    parser.add_argument("--query-index", type=int, default=0, help="把 memory bank 中第几个样本作为 query")
    parser.add_argument("--top-k", type=int, default=5, help="返回多少个检索结果")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 读取 npz memory bank。
    bank = np.load(args.memory, allow_pickle=False)
    domains = bank["domains"].astype(str)

    # 把 real/imag 重新合成 complex spectrum。
    spectra = bank["spectra_real"] + 1j * bank["spectra_imag"]

    query_index = int(args.query_index)
    query_domain = str(domains[query_index])
    query_domain_cfg = cfg["domains"][query_domain]

    # 根据 query 所属 domain 生成频率网格。
    freq = domain_frequency_grid(query_domain_cfg, int(cfg["response_stride"]))

    # 从配置读取 retrieval 权重。
    weights_cfg = cfg.get("retrieval", {})
    retriever = DomainAwareRetriever(
        summary_features=bank["summary_features"],
        domains=domains,
        params=bank["params"],
        spectra=spectra,
        domain_specs=cfg["domains"],
        weights=RetrievalWeights(
            summary_weight=float(weights_cfg.get("summary_weight", 0.70)),
            domain_weight=float(weights_cfg.get("domain_weight", 0.20)),
            constraint_weight=float(weights_cfg.get("constraint_weight", 0.10)),
        ),
    )

    # exclude_index=query_index 表示排除 query 自己，避免 top1 永远是自己。
    results = retriever.query(
        target_spectrum=spectra[query_index],
        target_frequency=freq,
        target_domain=query_domain,
        top_k=args.top_k,
        exclude_index=query_index,
    )

    print(json.dumps(
        {
            "query_index": query_index,
            "query_domain": query_domain,
            "top_k": results,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
