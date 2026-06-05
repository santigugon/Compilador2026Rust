from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def collect_attempt_metrics(experiment_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in experiment_dir.rglob("metrics.json"):
        try:
            rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def build_all_tables(experiment_dir: Path) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    rows = collect_attempt_metrics(experiment_dir)
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    outputs: dict[str, str] = {}

    parquet_path = experiment_dir / "attempts.parquet"
    df.to_parquet(parquet_path, index=False)
    outputs["attempts_parquet"] = str(parquet_path)

    csv_path = experiment_dir / "attempts.csv"
    df.to_csv(csv_path, index=False)
    outputs["attempts_csv"] = str(csv_path)

    if "quantization" in df.columns:
        agg_q = (
            df.groupby("quantization", dropna=False)
            .agg(
                attempts=("overall_passed", "count"),
                overall_pass_rate=("overall_passed", "mean"),
                compile_rate=("compiled", "mean"),
                correctness_rate=("correctness_passed", "mean"),
                mean_gen_latency_ms=("generation_latency_ms", "mean"),
            )
            .reset_index()
        )
        p = experiment_dir / "aggregate_by_quantization.csv"
        agg_q.to_csv(p, index=False)
        outputs["aggregate_by_quantization"] = str(p)

    if "role" in df.columns:
        agg_r = (
            df.groupby("role", dropna=False)
            .agg(
                attempts=("overall_passed", "count"),
                overall_pass_rate=("overall_passed", "mean"),
            )
            .reset_index()
        )
        p = experiment_dir / "aggregate_by_role.csv"
        agg_r.to_csv(p, index=False)
        outputs["aggregate_by_role"] = str(p)

    if "decoding_mode" in df.columns:
        agg_d = (
            df.groupby("decoding_mode", dropna=False)
            .agg(
                schema_valid_rate=("schema_valid", "mean"),
                compile_rate=("compiled", "mean"),
            )
            .reset_index()
        )
        p = experiment_dir / "aggregate_by_decoding_mode.csv"
        agg_d.to_csv(p, index=False)
        outputs["aggregate_by_decoding_mode"] = str(p)

    if "task_id" in df.columns:
        agg_t = (
            df.groupby("task_id", dropna=False)
            .agg(
                attempts=("overall_passed", "count"),
                correctness_rate=("correctness_passed", "mean"),
            )
            .reset_index()
        )
        p = experiment_dir / "aggregate_by_task.csv"
        agg_t.to_csv(p, index=False)
        outputs["aggregate_by_task"] = str(p)

    if {"quantization", "correctness_passed", "peak_gpu_memory_mb"}.issubset(df.columns):
        pareto = df.groupby("quantization", dropna=False).agg(
            correctness_rate=("correctness_passed", "mean"),
            peak_gpu_memory_mb=("peak_gpu_memory_mb", "mean"),
            mean_latency_ms=("generation_latency_ms", "mean"),
        )
        p = experiment_dir / "pareto_frontier.csv"
        pareto.reset_index().to_csv(p, index=False)
        outputs["pareto_frontier"] = str(p)

    fail = df.explode("failure_labels")
    if "failure_labels" in fail.columns:
        fc = (
            fail["failure_labels"]
            .dropna()
            .value_counts()
            .reset_index(name="count")
        )
        fc.columns = ["failure_label", "count"]
        p = experiment_dir / "failure_counts.csv"
        fc.to_csv(p, index=False)
        outputs["failure_counts"] = str(p)

    return outputs
