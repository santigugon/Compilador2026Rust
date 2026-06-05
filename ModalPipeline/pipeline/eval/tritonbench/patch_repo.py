from __future__ import annotations

import py_compile
from pathlib import Path


def _replace_line_prefix(path: Path, prefix: str, replacement: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    patched = [
        replacement if line.startswith(prefix) else line
        for line in lines
    ]
    path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def _fix_python311_fstrings(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'print(f"Above is call test for {path.split("/")[-1].replace(".jsonl", "")}")',
        'print(f"Above is call test for {path.split(\'/\')[-1].replace(\'.jsonl\', \'\')}")',
    )
    path.write_text(text, encoding="utf-8")


def _link_module(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def ensure_tritonbench_patched(repo_dir: str | Path) -> None:
    repo = Path(repo_dir)
    call_acc = repo / "EVAL" / "eval_T" / "0_call_acc.py"
    exe_acc = repo / "EVAL" / "eval_T" / "1_exe_acc.py"
    perf_runner = repo / "performance_metrics" / "perf_T" / "run_bench" / "multiprocess_gpu_run.py"

    if not call_acc.exists():
        raise FileNotFoundError(f"TritonBench call_acc script not found: {call_acc}")
    if not exe_acc.exists():
        raise FileNotFoundError(f"TritonBench exe_acc script not found: {exe_acc}")
    if not perf_runner.exists():
        raise FileNotFoundError(f"TritonBench perf runner not found: {perf_runner}")

    _replace_line_prefix(
        call_acc,
        "statis_path = ",
        f'statis_path = "{repo}/data/TritonBench_T_v1.jsonl"',
    )
    _replace_line_prefix(
        call_acc,
        "py_folder = ",
        f'py_folder = "{repo}/data/TritonBench_T_v1/"',
    )
    _replace_line_prefix(
        call_acc,
        "py_interpreter = ",
        "import sys; py_interpreter = sys.executable",
    )
    _fix_python311_fstrings(call_acc)

    _replace_line_prefix(
        exe_acc,
        "gold_folder = ",
        f'gold_folder = "{repo}/data/TritonBench_T_v1/"',
    )
    _replace_line_prefix(
        exe_acc,
        "py_interpreter = ",
        "import sys; py_interpreter = sys.executable",
    )

    _replace_line_prefix(perf_runner, "gpu_count = ", "gpu_count = 1")

    eval_dir = repo / "EVAL" / "eval_T"
    _link_module(call_acc, eval_dir / "call_acc.py")
    _link_module(exe_acc, eval_dir / "exe_acc.py")

    py_compile.compile(str(eval_dir / "call_acc.py"), doraise=True)
    py_compile.compile(str(eval_dir / "exe_acc.py"), doraise=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("repo_dir", help="Path to the TritonBench checkout")
    args = parser.parse_args()
    ensure_tritonbench_patched(args.repo_dir)


if __name__ == "__main__":
    main()
