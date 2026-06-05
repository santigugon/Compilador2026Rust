from __future__ import annotations

import subprocess
from pathlib import Path


def run_mini_triton_parser(
    parser_binary: Path,
    code: str,
    *,
    timeout_s: int = 30,
) -> dict:
    tmp = Path("/tmp/minitriton_check.mt")
    tmp.write_text(code, encoding="utf-8")

    try:
        proc = subprocess.run(
            [str(parser_binary), str(tmp)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        valid = "VALIDO" in stdout and proc.returncode == 0
        return {
            "valid": valid,
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "timeout",
        }
