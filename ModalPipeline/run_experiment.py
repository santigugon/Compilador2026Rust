#!/usr/bin/env python3
"""Launch Qwen3-Coder × TritonBench experiments on Modal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pipeline.env import load_project_dotenv, sync_modal_hf_secret_from_env

load_project_dotenv()

from pipeline.config_loader import (
    CONFIGS_DIR,
    _deep_merge,
    condition_id,
    effective_benchmark_limit,
    expand_matrix,
    load_experiment,
    load_hf_models,
    resolve_models,
)

# Run the full-precision baseline first, then descend through quantized variants.
_SERVE_ORDER = {
    "bf16_or_fp16": 0,
    "awq_8bit": 1,
    "gptq_int8": 2,
    "awq_4bit": 3,
    "gptq_4bit": 4,
}


def _model_for_run(model: dict, experiment: dict) -> dict:
    payload = dict(model)
    if experiment.get("vllm"):
        payload["vllm"] = _deep_merge(payload.get("vllm", {}), experiment["vllm"])
    return payload


def _sort_models_for_serve(models: list[dict]) -> list[dict]:
    return sorted(
        models,
        key=lambda m: (
            0 if m.get("is_baseline") else 1,
            _SERVE_ORDER.get(m.get("quantization", ""), 99),
        ),
    )


def _seed_dir(seed: int | str) -> str:
    return f"seed_{int(seed):03d}"


def _merge_tree_preserving_local(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            _merge_tree_preserving_local(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def export_results_to_local(experiment_id: str, seeds: list[int | str]) -> None:
    unique_seeds = sorted({int(seed) for seed in seeds})
    for seed in unique_seeds:
        local_dir = PACKAGE_ROOT / "local_results" / experiment_id / _seed_dir(seed)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{experiment_id}-export-") as tmp:
            tmp_dir = Path(tmp)
            proc = subprocess.run(
                [
                    "modal",
                    "volume",
                    "get",
                    "qwen3-coder-results",
                    f"/{experiment_id}",
                    str(tmp_dir),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                _merge_tree_preserving_local(tmp_dir, local_dir)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            print(f"Warning: failed to auto-export results for seed {seed}: {err}")
            continue
        print(f"Exported Modal results to {local_dir}")


def _finish_one_modal_call(
    in_flight: list[dict],
    *,
    experiment_id: str,
    auto_export: bool,
    errors: list[tuple[dict, BaseException]],
    fail_fast: bool,
) -> None:
    job = in_flight.pop(0)
    try:
        out = job["call"].get()
        print(f"Completed {job['model_name']} / {job['condition_id']}")
        job["result"] = out
    except BaseException as exc:
        print(f"Failed {job['model_name']} / {job['condition_id']}: {exc}")
        errors.append((job, exc))
        try:
            modal_app.record_failed_condition.remote(
                experiment_id,
                job["model_name"],
                job["condition_id"],
                int(job["seed"]),
                str(exc),
            )
        except Exception as record_exc:
            print(
                "Warning: failed to record remote failure checkpoint for "
                f"{job['condition_id']}: {record_exc}"
            )
    finally:
        if auto_export:
            export_results_to_local(experiment_id, [job["seed"]])
    if fail_fast:
        _raise_if_modal_errors(errors)


def _raise_if_modal_errors(errors: list[tuple[dict, BaseException]]) -> None:
    if not errors:
        return
    job, exc = errors[0]
    raise RuntimeError(
        f"{len(errors)} Modal job(s) failed; first failure was "
        f"{job['model_name']} / {job['condition_id']}"
    ) from exc


from pipeline.logging.artifacts import git_commit_hash

import app as modal_app  # noqa: E402 — registers Modal functions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Qwen3-Coder TritonBench experiment")
    p.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "qwen3_coder_experiment.yaml",
        help="Experiment YAML path",
    )
    p.add_argument(
        "--hf-models",
        type=Path,
        default=CONFIGS_DIR / "hf_models.yaml",
        help="HF models YAML path",
    )
    p.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model names to run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without calling Modal",
    )
    p.add_argument(
        "--sync-secrets",
        action="store_true",
        help="Push HF_TOKEN from .env to Modal secret before running",
    )
    p.add_argument(
        "--no-sync-secrets",
        action="store_true",
        help="Skip .env → Modal secret sync even if SYNC_SECRETS_ON_RUN=1",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Run locally (requires GPU + TritonBench); no Modal",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip conditions with completed checkpoints in the Modal results volume",
    )
    p.add_argument(
        "--no-auto-export",
        action="store_true",
        help="Do not automatically download Modal results into local_results after jobs",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("MODAL_JOB_CONCURRENCY", "1")),
        help="Maximum number of generation/evaluation Modal jobs to run at once",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop before finalization if any Modal job fails",
    )
    return p.parse_args()


def save_experiment_snapshot(experiment: dict, hf_path: Path, config_path: Path) -> None:
    exp_id = experiment["experiment_id"]
    out_dir = PACKAGE_ROOT / "local_results" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")
    shutil.copy(hf_path, out_dir / "hf_models.yaml")
    env = {"git_commit": git_commit_hash()}
    (out_dir / "environment.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    experiment = load_experiment(args.config)
    hf_config = load_hf_models(args.hf_models)
    hf_defaults = hf_config.get("defaults", {})
    models = resolve_models(hf_config)

    if args.models:
        names = set(args.models)
        models = [m for m in models if m["name"] in names]

    if not experiment.get("preserve_model_order", False):
        models = _sort_models_for_serve(models)

    conditions = expand_matrix(experiment)
    experiment_id = experiment["experiment_id"]

    save_experiment_snapshot(experiment, args.hf_models, args.config)
    auto_export = not args.no_auto_export
    resume = not args.no_resume
    concurrency = max(1, int(args.concurrency))

    jobs: list[dict] = []
    for model in models:
        gpu = model.get("gpu_serve") or hf_defaults.get("gpu_serve", "A10G")
        for cond in conditions:
            cid = condition_id(model["name"], cond)
            jobs.append(
                {
                    "model": model["name"],
                    "hf_id": model["hf_id"],
                    "quantization": model.get("quantization"),
                    "gpu": gpu,
                    "benchmark_limit": effective_benchmark_limit(
                        model,
                        experiment,
                        hf_defaults,
                    ),
                    "condition_id": cid,
                    "condition": cond,
                }
            )

    print(f"experiment_id={experiment_id} jobs={len(jobs)} models={len(models)}")
    for j in jobs:
        print(
            f"  - {j['model']} / gpu={j['gpu']} / "
            f"limit={j['benchmark_limit']} / {j['condition_id']}"
        )

    if args.dry_run:
        return

    if args.local:
        from pipeline.runner_core import run_condition_local

        tritonbench_repo = Path(
            experiment.get("benchmark", {}).get(
                "tritonbench_local_path", "/opt/TritonBench"
            )
        )
        parser = PACKAGE_ROOT.parent / "Compiler/target/release/mini_triton_parser"
        results_root = PACKAGE_ROOT / "local_results"
        for model in models:
            model_payload = _model_for_run(model, experiment)
            for cond in conditions:
                run_condition_local(
                    model=model_payload,
                    condition=cond,
                    experiment={**experiment, "_hf_defaults": hf_defaults},
                    results_root=results_root,
                    model_cache=PACKAGE_ROOT / ".model_cache",
                    tritonbench_repo=tritonbench_repo,
                    parser_binary=parser if parser.exists() else None,
                )
        from pipeline.aggregate.build_tables import build_all_tables

        build_all_tables(results_root / experiment_id)
        return

    should_sync = args.sync_secrets or (
        not args.no_sync_secrets
        and os.environ.get("SYNC_SECRETS_ON_RUN", "1").strip() not in ("0", "false", "no")
    )
    if should_sync:
        print("Syncing HF_TOKEN from .env to Modal secret...")
        sync_modal_hf_secret_from_env(force=False)
    else:
        print(
            "Skipping secret sync (use --sync-secrets or set SYNC_SECRETS_ON_RUN=1 in .env)"
        )

    in_flight: list[dict] = []
    errors: list[tuple[dict, BaseException]] = []

    for model in models:
        gpu = model.get("gpu_serve") or hf_defaults.get("gpu_serve", "A10G")
        model_payload = _model_for_run(model, experiment)
        for cond in conditions:
            cid = condition_id(model["name"], cond)
            if resume and modal_app.condition_checkpoint_exists.remote(experiment_id, cid):
                print(f"Skipping completed checkpoint: {model['name']} / {cid}")
                if auto_export:
                    export_results_to_local(experiment_id, [cond["seed"]])
                continue
            while len(in_flight) >= concurrency:
                _finish_one_modal_call(
                    in_flight,
                    experiment_id=experiment_id,
                    auto_export=auto_export,
                    errors=errors,
                    fail_fast=args.fail_fast,
                )

            print(f"Launching {model['name']} on {gpu} / {cid} ...")
            call = modal_app.run_generation_condition.with_options(gpu=gpu).spawn(
                model_payload,
                cond,
                experiment,
                hf_defaults,
            )
            in_flight.append(
                {
                    "call": call,
                    "model_name": model["name"],
                    "condition_id": cid,
                    "seed": cond["seed"],
                }
            )

    while in_flight:
        _finish_one_modal_call(
            in_flight,
            experiment_id=experiment_id,
            auto_export=auto_export,
            errors=errors,
            fail_fast=args.fail_fast,
        )

    manifest = modal_app.finalize_experiment.remote(experiment_id)
    print(json.dumps(manifest, indent=2))
    if auto_export:
        export_results_to_local(experiment_id, [cond["seed"] for cond in conditions])
    if errors:
        print(f"Completed with {len(errors)} failed Modal job(s):")
        for job, exc in errors:
            print(f"  - {job['model_name']} / {job['condition_id']}: {exc}")


@modal_app.app.local_entrypoint()
def modal_entrypoint(
    config: str = "configs/qwen3_coder_experiment.yaml",
    hf_models: str = "configs/hf_models.yaml",
    models: str = "",
    dry_run: bool = False,
    no_resume: bool = False,
    no_auto_export: bool = False,
    concurrency: int = 1,
    fail_fast: bool = False,
) -> None:
    argv = [
        "--config",
        config,
        "--hf-models",
        hf_models,
    ]
    if models.strip():
        argv.extend(["--models", *models.split()])
    if dry_run:
        argv.append("--dry-run")
    if no_resume:
        argv.append("--no-resume")
    if no_auto_export:
        argv.append("--no-auto-export")
    argv.extend(["--concurrency", str(concurrency)])
    if fail_fast:
        argv.append("--fail-fast")
    sys.argv = ["run_experiment.py", *argv]
    main()


if __name__ == "__main__":
    main()
