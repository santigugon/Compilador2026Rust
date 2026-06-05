#!/usr/bin/env python3
"""Pick the most compressed model that still meets quality thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

QUANT_ORDER = [
    "bf16_or_fp16",
    "awq_8bit",
    "gptq_int8",
    "awq_4bit",
    "gptq_4bit",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("experiment_dir", type=Path)
    p.add_argument("--min-exe-acc", type=float, default=0.3)
    p.add_argument("--min-compile-rate", type=float, default=0.3)
    args = p.parse_args()

    parquet = args.experiment_dir / "attempts.parquet"
    if not parquet.exists():
        print(json.dumps({"error": "attempts.parquet not found"}))
        return

    df = pd.read_parquet(parquet)
    agg = (
        df.groupby(["model_name", "quantization"], dropna=False)
        .agg(
            exe_acc=("correctness_passed", "mean"),
            compile_rate=("compiled", "mean"),
        )
        .reset_index()
    )

    candidates = []
    for quant in reversed(QUANT_ORDER):
        sub = agg[agg["quantization"] == quant]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if (
            row["exe_acc"] >= args.min_exe_acc
            and row["compile_rate"] >= args.min_compile_rate
        ):
            candidates.append(row.to_dict())

    result = {
        "recommended": candidates[0] if candidates else None,
        "all_candidates": candidates,
    }
    out = args.experiment_dir / "recommendation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
