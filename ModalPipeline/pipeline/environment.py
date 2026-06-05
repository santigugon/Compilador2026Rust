from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any


def collect_environment(
    *,
    modal_app_name: str | None = None,
    modal_function: str | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "modal_app_name": modal_app_name,
        "modal_function": modal_function,
    }

    for pkg in ("vllm", "torch", "triton", "xgrammar", "modal"):
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            env[f"{pkg}_version"] = None

    try:
        env["cuda_version"] = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        env["cuda_version"] = os.environ.get("CUDA_VERSION")

    try:
        from pipeline.logging.artifacts import git_commit_hash

        env["git_commit"] = git_commit_hash()
    except Exception:
        env["git_commit"] = None

    return env
