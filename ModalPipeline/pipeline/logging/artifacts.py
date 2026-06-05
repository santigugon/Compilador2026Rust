from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


class ArtifactWriter:
    def __init__(self, results_root: Path, experiment_id: str) -> None:
        self.results_root = Path(results_root)
        self.experiment_id = experiment_id
        self.experiment_dir = self.results_root / experiment_id

    def attempt_dir(
        self,
        model_name: str,
        quantization: str,
        output_profile: str,
        decoding_mode: str,
        task_idx: int,
        seed: int,
        attempt: int,
    ) -> Path:
        path = (
            self.experiment_dir
            / "models"
            / model_name
            / quantization
            / output_profile
            / decoding_mode
            / f"task_{task_idx:03d}"
            / f"seed_{seed:03d}"
            / f"attempt_{attempt:03d}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def write_json(self, path: Path, data: Any) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(path)

    def save_attempt(
        self,
        *,
        base_dir: Path,
        metadata: dict[str, Any],
        metrics: dict[str, Any],
        failure: dict[str, Any] | None,
        prompt: str,
        raw_output: str,
        parsed_output: dict[str, Any] | None = None,
        extracted_code: str | None = None,
        test_code: str | None = None,
        compile_stdout: str = "",
        compile_stderr: str = "",
        runtime_stdout: str = "",
        runtime_stderr: str = "",
        environment: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        paths["metadata"] = self.write_json(base_dir / "metadata.json", metadata)
        paths["metrics"] = self.write_json(base_dir / "metrics.json", metrics)
        if failure is not None:
            paths["failure"] = self.write_json(base_dir / "failure.json", failure)
        paths["prompt"] = self.write_text(base_dir / "prompt.txt", prompt)
        paths["raw_output"] = self.write_text(base_dir / "raw_output.txt", raw_output)
        if parsed_output is not None:
            paths["parsed_output"] = self.write_json(
                base_dir / "parsed_output.json", parsed_output
            )
        if extracted_code is not None:
            paths["minitriton_code"] = self.write_text(
                base_dir / "extracted_code.py", extracted_code
            )
        if test_code is not None:
            paths["test_code"] = self.write_text(base_dir / "test_code.py", test_code)
        paths["compile_stdout"] = self.write_text(
            base_dir / "compile_stdout.txt", compile_stdout
        )
        paths["compile_stderr"] = self.write_text(
            base_dir / "compile_stderr.txt", compile_stderr
        )
        paths["runtime_stdout"] = self.write_text(
            base_dir / "runtime_stdout.txt", runtime_stdout
        )
        paths["runtime_stderr"] = self.write_text(
            base_dir / "runtime_stderr.txt", runtime_stderr
        )
        if environment:
            paths["environment"] = self.write_json(
                base_dir / "environment.json", environment
            )
        return paths


def new_run_id() -> str:
    return str(uuid4())


def build_attempt_metrics(
    *,
    run_id: str,
    experiment_id: str,
    model: dict[str, Any],
    condition: dict[str, Any],
    task_id: str,
    task_idx: int,
    seed: int,
    attempt: int,
    repair_round: int,
    artifact_paths: dict[str, str],
    gen_stats: dict[str, Any],
    eval_stats: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "timestamp_utc": utc_now_iso(),
        "model_name": model["name"],
        "hf_model_id": model["hf_id"],
        "base_model": model.get("base_model"),
        "model_revision": model.get("model_revision"),
        "quantization": model.get("quantization"),
        "quantization_name": model["name"],
        "quantization_method": model.get("vllm", {}).get("quantization"),
        "role": model.get("role"),
        "backend": model.get("backend"),
        "weight_dtype": model.get("quantization"),
        "decoding_mode": condition["decoding_mode"],
        "xgrammar_enabled": condition.get("xgrammar_enabled", False),
        "grammar_version": condition.get("grammar_version"),
        "prompt_template_version": condition.get("prompt_template_version"),
        "output_profile": condition["output_profile"],
        "task_id": task_id,
        "task_idx": task_idx,
        "seed": seed,
        "attempt": attempt,
        "repair_round": repair_round,
        "temperature": condition.get("temperature", 0.0),
        "top_p": condition.get("top_p", 1.0),
        "max_output_tokens": condition.get("max_output_tokens", 2048),
        "prompt_tokens": gen_stats.get("prompt_tokens", 0),
        "completion_tokens": gen_stats.get("completion_tokens", 0),
        "generation_latency_ms": gen_stats.get("generation_latency_ms", 0),
        "tokens_per_second": gen_stats.get("tokens_per_second", 0),
        "raw_output_path": artifact_paths.get("raw_output"),
        "parsed_output_path": artifact_paths.get("parsed_output"),
        "minitriton_code_path": artifact_paths.get("minitriton_code")
        or artifact_paths.get("extracted_code"),
        "test_code_path": artifact_paths.get("test_code"),
        "structure_valid": summary.get("structure_valid", False),
        "schema_valid": summary.get("schema_valid", False),
        "code_extracted": summary.get("code_extracted", False),
        "compiled": summary.get("compiled", False),
        "compile_duration_ms": eval_stats.get("compile_duration_ms", 0),
        "compile_stdout_path": artifact_paths.get("compile_stdout"),
        "compile_stderr_path": artifact_paths.get("compile_stderr"),
        "ran_successfully": summary.get("ran_successfully", False),
        "runtime_duration_ms": eval_stats.get("runtime_duration_ms", 0),
        "runtime_stdout_path": artifact_paths.get("runtime_stdout"),
        "runtime_stderr_path": artifact_paths.get("runtime_stderr"),
        "correctness_passed": summary.get("correctness_passed", False),
        "max_abs_error": eval_stats.get("max_abs_error"),
        "max_rel_error": eval_stats.get("max_rel_error"),
        "mean_abs_error": eval_stats.get("mean_abs_error"),
        "rtol": eval_stats.get("rtol"),
        "atol": eval_stats.get("atol"),
        "mean_latency_ms": eval_stats.get("mean_latency_ms"),
        "median_latency_ms": eval_stats.get("median_latency_ms"),
        "p95_latency_ms": eval_stats.get("p95_latency_ms"),
        "p99_latency_ms": eval_stats.get("p99_latency_ms"),
        "throughput": eval_stats.get("throughput"),
        "pytorch_baseline_latency_ms": eval_stats.get("pytorch_baseline_latency_ms"),
        "speedup_vs_pytorch": eval_stats.get("speedup_vs_pytorch"),
        "gpu_type": eval_stats.get("gpu_type"),
        "gpu_count": eval_stats.get("gpu_count", 1),
        "peak_gpu_memory_mb": eval_stats.get("peak_gpu_memory_mb"),
        "cpu_memory_mb": eval_stats.get("cpu_memory_mb"),
        "failure_stage": summary.get("failure_stage"),
        "failure_labels": summary.get("failure_labels", []),
        "overall_passed": summary.get("overall_passed", False),
        "notes": summary.get("notes", ""),
    }
