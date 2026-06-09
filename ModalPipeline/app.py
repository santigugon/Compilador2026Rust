"""Modal app: vLLM generation + TritonBench eval on Modal GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from pipeline.secrets_config import modal_hf_secret_name

APP_NAME = "qwen3-coder-tritonbench"
PACKAGE_ROOT = Path(__file__).resolve().parent
COMPILER_ROOT = PACKAGE_ROOT.parent / "Compiler"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"
REPO_DIR = "/opt/TritonBench"

RESULTS_MOUNT = "/results"
MODEL_CACHE_MOUNT = "/models"

_MODAL_IGNORE = [
    ".venv",
    "local_results",
    ".model_cache",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
]

_tritonbench_clone = [
    f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}",
]

app = modal.App(APP_NAME)

volume_models = modal.Volume.from_name("qwen3-model-cache", create_if_missing=True)
volume_results = modal.Volume.from_name("qwen3-coder-results", create_if_missing=True)

hf_secret = modal.Secret.from_name(modal_hf_secret_name())

_package_env = {
    "PYTHONPATH": "/root/ModalPipeline",
    "MODAL_PIPELINE_ROOT": "/root/ModalPipeline",
    "COMPILER_ROOT": str(COMPILER_ROOT),
    "TRITONBENCH_REPO_DIR": REPO_DIR,
    "HF_HOME": MODEL_CACHE_MOUNT,
    "HUGGINGFACE_HUB_CACHE": MODEL_CACHE_MOUNT,
    "RESULTS_ROOT": RESULTS_MOUNT,
}

# vLLM + FlashInfer JIT sampling needs nvcc (not present on debian_slim).
_CUDA_SERVE_BASE = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.11",
)
_SERVE_ENV = {
    **_package_env,
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
}


def _pipeline_image(
    *,
    base: modal.Image,
    requirements_file: str,
    env: dict[str, str],
) -> modal.Image:
    """All pip/apt/run steps before add_local_dir; env set before mount."""
    return (
        base.apt_install("git", "curl", "build-essential")
        .run_commands(*_tritonbench_clone)
        .pip_install_from_requirements(str(PACKAGE_ROOT / requirements_file))
        .env(env)
        .add_local_dir(
            str(PACKAGE_ROOT),
            remote_path="/root/ModalPipeline",
            ignore=_MODAL_IGNORE,
        )
    )


image_serve = _pipeline_image(
    base=_CUDA_SERVE_BASE,
    requirements_file="requirements-serve.txt",
    env=_SERVE_ENV,
)
image_eval = _pipeline_image(
    base=modal.Image.debian_slim(python_version="3.11"),
    requirements_file="requirements-eval.txt",
    env=_package_env,
)


def _gpu_for_model(model_cfg: dict, hf_defaults: dict) -> str:
    return model_cfg.get("gpu_serve") or hf_defaults.get("gpu_serve", "A10G")


@app.function(
    image=image_serve,
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
    volumes={
        MODEL_CACHE_MOUNT: volume_models,
        RESULTS_MOUNT: volume_results,
    },
    secrets=[hf_secret],
)
def run_generation_condition(
    model_cfg: dict,
    condition: dict,
    experiment: dict,
    hf_defaults: dict,
) -> dict:
    import sys

    sys.path.insert(0, "/root/ModalPipeline")

    from pipeline.runner_core import run_condition_local

    experiment = dict(experiment)
    experiment["_hf_defaults"] = hf_defaults
    parser_bin = Path("/root/Compiler/target/release/mini_triton_parser")
    if not parser_bin.exists():
        parser_bin = None

    out = run_condition_local(
        model=model_cfg,
        condition=condition,
        experiment=experiment,
        results_root=Path(RESULTS_MOUNT),
        model_cache=Path(MODEL_CACHE_MOUNT),
        tritonbench_repo=Path(REPO_DIR),
        parser_binary=parser_bin,
    )

    checkpoint_dir = Path(RESULTS_MOUNT) / experiment["experiment_id"] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{out['condition_id']}.json"
    eval_summary = out.get("eval_summary") or {}
    total_predictions = int(eval_summary.get("total_predictions") or 0)
    passed_predictions = int(
        (eval_summary.get("phase2_exec_acc") or {}).get("passed") or 0
    )
    status = (
        "completed_with_failures"
        if total_predictions and passed_predictions < total_predictions
        else "completed"
    )
    checkpoint_path.write_text(
        json.dumps(
            {
                "status": status,
                "model_name": model_cfg["name"],
                "condition_id": out["condition_id"],
                "seed": condition.get("seed"),
                "predictions_path": out.get("predictions_path"),
                "attempt_count": out.get("attempt_count"),
                "total_predictions": total_predictions,
                "passed_predictions": passed_predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    volume_results.commit()
    volume_models.commit()
    return out


@app.function(
    image=image_eval,
    cpu=1,
    timeout=60 * 5,
    volumes={RESULTS_MOUNT: volume_results},
)
def condition_checkpoint_exists(experiment_id: str, condition_id: str) -> bool:
    checkpoint = Path(RESULTS_MOUNT) / experiment_id / "checkpoints" / f"{condition_id}.json"
    return checkpoint.exists()


@app.function(
    image=image_eval,
    cpu=1,
    timeout=60 * 5,
    volumes={RESULTS_MOUNT: volume_results},
)
def record_failed_condition(
    experiment_id: str,
    model_name: str,
    condition_id: str,
    seed: int,
    error: str,
) -> dict:
    failure_dir = Path(RESULTS_MOUNT) / experiment_id / "failed_checkpoints"
    failure_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "model_name": model_name,
        "condition_id": condition_id,
        "seed": seed,
        "error": error,
    }
    path = failure_dir / f"{condition_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    volume_results.commit()
    return {"path": str(path), **payload}


@app.function(
    image=image_eval,
    gpu="T4",
    timeout=60 * 60 * 6,
    volumes={RESULTS_MOUNT: volume_results},
    secrets=[hf_secret],
)
def run_tritonbench_eval_only(
    predictions_rel_path: str,
    experiment_id: str,
    model_name: str,
    condition_id: str,
    run_perf: bool = True,
) -> dict:
    import sys

    sys.path.insert(0, "/root/ModalPipeline")

    from pipeline.eval.tritonbench.runner import TritonBenchRunner

    pred_path = Path(RESULTS_MOUNT) / predictions_rel_path
    out_dir = (
        Path(RESULTS_MOUNT) / experiment_id / "eval" / model_name / condition_id
    )
    runner = TritonBenchRunner(REPO_DIR)
    summary = runner.evaluate_predictions(pred_path, out_dir, run_perf=run_perf)
    volume_results.commit()
    return summary


@app.function(
    image=image_eval,
    cpu=2,
    timeout=60 * 30,
    volumes={RESULTS_MOUNT: volume_results},
)
def finalize_experiment(experiment_id: str) -> dict:
    import sys

    sys.path.insert(0, "/root/ModalPipeline")

    from pipeline.aggregate.build_tables import build_all_tables

    exp_dir = Path(RESULTS_MOUNT) / experiment_id
    tables = build_all_tables(exp_dir)
    manifest = {"experiment_id": experiment_id, "tables": tables}
    (exp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    volume_results.commit()
    return manifest


@app.function(
    image=image_serve,
    cpu=2,
    volumes={MODEL_CACHE_MOUNT: volume_models},
    secrets=[hf_secret],
)
def prefetch_model(hf_id: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        hf_id,
        cache_dir=MODEL_CACHE_MOUNT,
        token=os.environ.get("HF_TOKEN") or None,
    )
    volume_models.commit()
    return path
