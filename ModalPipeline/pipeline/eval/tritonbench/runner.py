from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.eval.tritonbench.patch_repo import ensure_tritonbench_patched


def load_alpaca_dataset(repo_dir: Path, dataset: str) -> list[dict]:
    assert dataset in ("simp", "comp")
    path = repo_dir / f"data/TritonBench_T_{dataset}_alpac_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TritonBenchRunner:
    def __init__(self, repo_dir: str | Path) -> None:
        self.repo_dir = Path(repo_dir)
        self.eval_dir = self.repo_dir / "EVAL" / "eval_T"
        self.perf_root = self.repo_dir / "performance_metrics" / "perf_T"

    def evaluate_predictions(
        self,
        predictions_path: Path,
        output_dir: Path,
        *,
        gpus: list[int] | None = None,
        run_perf: bool = True,
    ) -> dict[str, Any]:
        gpus = gpus or [0]
        output_dir.mkdir(parents=True, exist_ok=True)
        call_acc_dir = output_dir / "call_acc"
        perf_results_dir = output_dir / "perf_results"

        if call_acc_dir.exists():
            shutil.rmtree(call_acc_dir)
        if perf_results_dir.exists():
            shutil.rmtree(perf_results_dir)

        ensure_tritonbench_patched(self.repo_dir)

        if str(self.eval_dir) not in sys.path:
            sys.path.insert(0, str(self.eval_dir))
        os.environ["PYTHONPATH"] = (
            str(self.eval_dir)
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        )

        import call_acc  # type: ignore
        import exe_acc  # type: ignore

        total = sum(1 for _ in predictions_path.open())
        t0 = time.perf_counter()

        call_acc.call_4file(str(predictions_path), str(call_acc_dir), gpus=gpus)
        call_duration_ms = (time.perf_counter() - t0) * 1000

        call_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))

        if call_survivors:
            exe_acc.execute_4folder(str(call_acc_dir), gpus=gpus)

        exec_survivors = sorted(p.name for p in call_acc_dir.glob("*.py"))

        per_op: dict[str, dict[str, Any]] = {}
        for name in call_survivors:
            per_op.setdefault(name, {})["call_acc_passed"] = True
        for name in exec_survivors:
            per_op.setdefault(name, {})["exe_acc_passed"] = True

        speedup = None
        eff_summary = "skipped"
        perf_duration_ms = 0.0

        if run_perf and exec_survivors:
            subprocess.run(
                [
                    sys.executable,
                    "run_bench/write_file.py",
                    "--input_folder_path",
                    str(call_acc_dir),
                    "--results_path",
                    str(perf_results_dir),
                ],
                cwd=str(self.perf_root),
                check=True,
            )
            subprocess.run(
                [sys.executable, "run_bench/multiprocess_gpu_run.py"],
                cwd=str(self.perf_root),
                check=True,
            )
            t_perf = time.perf_counter()
            eff = subprocess.run(
                [
                    sys.executable,
                    "2_efficiency.py",
                    "--gen_folder",
                    str(perf_results_dir),
                ],
                cwd=str(self.eval_dir),
                capture_output=True,
                text=True,
            )
            perf_duration_ms = (time.perf_counter() - t_perf) * 1000
            eff_summary = eff.stdout
            for line in eff.stdout.splitlines():
                if line.startswith("speed up:"):
                    try:
                        speedup = float(line.split(":")[1].strip())
                    except Exception:
                        pass

        return {
            "total_predictions": total,
            "phase1_call_acc": {
                "passed": len(call_survivors),
                "rate": round(100 * len(call_survivors) / total, 2) if total else 0,
                "duration_ms": call_duration_ms,
            },
            "phase2_exec_acc": {
                "passed": len(exec_survivors),
                "rate": round(100 * len(exec_survivors) / total, 2) if total else 0,
            },
            "phase3_efficiency": {
                "speedup_vs_pytorch": speedup,
                "raw_output_tail": eff_summary[-2000:],
                "duration_ms": perf_duration_ms,
            },
            "per_op": per_op,
            "call_acc_dir": str(call_acc_dir),
            "perf_results_dir": str(perf_results_dir),
        }
