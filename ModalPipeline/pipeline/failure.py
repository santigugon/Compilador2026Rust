from __future__ import annotations

from typing import Any


FAILURE_LABELS = [
    "invalid_json",
    "schema_violation",
    "missing_minitriton_code",
    "prose_outside_json",
    "markdown_fence_pollution",
    "invalid_minitriton_syntax",
    "hallucinated_minitriton_api",
    "missing_kernel_entrypoint",
    "compile_error",
    "runtime_error",
    "wrong_output_shape",
    "dtype_mismatch",
    "numerical_mismatch",
    "timeout",
    "memory_error",
    "performance_regression",
    "nondeterministic_result",
    "excessive_token_usage",
    "infrastructure_failure",
]


def classify_generation(
    *,
    output_profile: str,
    raw_output: str,
    parsed: dict | None,
    extracted_code: str | None,
    extraction_errors: list[str],
    schema_valid: bool,
    structure_valid: bool,
) -> tuple[str, list[str]]:
    labels: list[str] = []

    if output_profile == "minitriton_json":
        if parsed is None:
            labels.append("invalid_json")
            return "parse", labels
        if not schema_valid:
            labels.append("schema_violation")
            return "schema", labels
        if not extracted_code:
            labels.append("missing_minitriton_code")
            return "extract", labels
        if not structure_valid:
            labels.append("invalid_minitriton_syntax")
            return "structure", labels
        return "ok", labels

    if not extracted_code:
        if "missing_python_fence" in extraction_errors:
            labels.append("markdown_fence_pollution")
        else:
            labels.append("missing_kernel_entrypoint")
        return "extract", labels

    return "ok", labels


def classify_eval(
    *,
    call_acc_passed: bool,
    exe_acc_passed: bool,
    performance_passed: bool,
) -> tuple[str, list[str]]:
    labels: list[str] = []
    if not call_acc_passed:
        labels.append("compile_error")
        return "compile", labels
    if not exe_acc_passed:
        labels.append("numerical_mismatch")
        return "correctness", labels
    if not performance_passed:
        labels.append("performance_regression")
        return "performance", labels
    return "ok", labels


def build_summary(
    *,
    output_profile: str,
    gen_stage: str,
    gen_labels: list[str],
    eval_result: dict[str, Any] | None,
    op_passed: bool | None = None,
) -> dict[str, Any]:
    if gen_stage != "ok":
        return {
            "structure_valid": False,
            "schema_valid": False,
            "code_extracted": False,
            "compiled": False,
            "ran_successfully": False,
            "correctness_passed": False,
            "performance_passed": False,
            "overall_passed": False,
            "failure_stage": gen_stage,
            "failure_labels": gen_labels,
            "notes": "",
        }

    structure_valid = True
    schema_valid = output_profile != "minitriton_json"
    code_extracted = True

    compiled = False
    ran = False
    correctness = False
    performance = False
    labels = list(gen_labels)
    stage = "generation"

    if eval_result:
        compiled = op_passed if op_passed is not None else False
        ran = compiled
        correctness = eval_result.get("exe_acc_passed", False)
        performance = eval_result.get("perf_passed", True)
        eval_stage, eval_labels = classify_eval(
            call_acc_passed=compiled,
            exe_acc_passed=correctness,
            performance_passed=performance,
        )
        if eval_stage != "ok":
            stage = eval_stage
            labels.extend(eval_labels)

    overall = (
        structure_valid
        and code_extracted
        and (schema_valid or output_profile != "minitriton_json")
        and compiled
        and correctness
        and performance
    )

    return {
        "structure_valid": structure_valid,
        "schema_valid": schema_valid,
        "code_extracted": code_extracted,
        "compiled": compiled,
        "ran_successfully": ran,
        "correctness_passed": correctness,
        "performance_passed": performance,
        "overall_passed": overall,
        "failure_stage": stage if not overall else "none",
        "failure_labels": labels if not overall else [],
        "notes": "",
    }
