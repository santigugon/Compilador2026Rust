from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jsonschema

from pipeline.config_loader import (
    condition_id,
    effective_benchmark_limit,
    expand_matrix,
)
from pipeline.eval.minitriton.checker import run_mini_triton_parser
from pipeline.eval.tritonbench.runner import TritonBenchRunner, load_alpaca_dataset
from pipeline.failure import build_summary, classify_generation
from pipeline.generate.extract import extract_python_code, parse_json_response
from pipeline.generate.prompts import build_messages
from pipeline.generate.grammar import schema_for_profile
from pipeline.environment import collect_environment
from pipeline.logging.artifacts import (
    ArtifactWriter,
    build_attempt_metrics,
    new_run_id,
)
from pipeline.serve.vllm_engine import VllmGenerator


def get_model_revision(hf_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(hf_id)
        return info.sha if info else None
    except Exception:
        return None


def process_generation_output(
    *,
    output_profile: str,
    raw_output: str,
    instruction: str,
) -> dict[str, Any]:
    if output_profile == "tritonbench_python":
        parsed, json_errors = parse_json_response(raw_output)
        if parsed and isinstance(parsed.get("predict"), str):
            code = parsed["predict"]
            gen_stage, labels = classify_generation(
                output_profile=output_profile,
                raw_output=raw_output,
                parsed=parsed,
                extracted_code=code,
                extraction_errors=[],
                schema_valid=True,
                structure_valid=bool(code),
            )
            return {
                "parsed": parsed,
                "predict": code,
                "test_code": None,
                "extraction_errors": [],
                "schema_valid": True,
                "gen_stage": gen_stage,
                "gen_labels": labels,
                "record": {"instruction": instruction, "predict": code},
            }
        if parsed is not None:
            code = parsed.get("predict") if isinstance(parsed.get("predict"), str) else ""
            gen_stage, labels = classify_generation(
                output_profile=output_profile,
                raw_output=raw_output,
                parsed=parsed,
                extracted_code=code,
                extraction_errors=["missing_predict_field"],
                schema_valid=False,
                structure_valid=False,
            )
            return {
                "parsed": parsed,
                "predict": code,
                "test_code": None,
                "extraction_errors": ["missing_predict_field"],
                "schema_valid": False,
                "gen_stage": gen_stage,
                "gen_labels": labels,
                "record": {"instruction": instruction, "predict": code},
            }

        code, errors = extract_python_code(raw_output)
        errors = errors if errors else []
        if json_errors and raw_output.strip().startswith(("{", "```json")):
            errors.extend(json_errors)
        gen_stage, labels = classify_generation(
            output_profile=output_profile,
            raw_output=raw_output,
            parsed=None,
            extracted_code=code,
            extraction_errors=errors,
            schema_valid=True,
            structure_valid=bool(code),
        )
        predict = code or ""
        return {
            "parsed": None,
            "predict": predict,
            "test_code": None,
            "extraction_errors": errors,
            "schema_valid": True,
            "gen_stage": gen_stage,
            "gen_labels": labels,
            "record": {"instruction": instruction, "predict": predict},
        }

    if output_profile == "minitriton_json":
        parsed, errors = parse_json_response(raw_output)
        schema = schema_for_profile(output_profile)
        schema_valid = False
        extracted = None
        test_code = None
        if parsed:
            try:
                jsonschema.validate(parsed, schema)
                schema_valid = True
            except jsonschema.ValidationError:
                errors.append("schema_violation")
            extracted = parsed.get("minitriton_code") if parsed else None
            test_code = parsed.get("test_code") if parsed else None
        gen_stage, labels = classify_generation(
            output_profile=output_profile,
            raw_output=raw_output,
            parsed=parsed,
            extracted_code=extracted,
            extraction_errors=errors,
            schema_valid=schema_valid,
            structure_valid=False,
        )
        predict = extracted or ""
        return {
            "parsed": parsed,
            "predict": predict,
            "test_code": test_code,
            "extraction_errors": errors,
            "schema_valid": schema_valid,
            "gen_stage": gen_stage,
            "gen_labels": labels,
            "record": {"instruction": instruction, "predict": raw_output},
        }

    code, errors = extract_python_code(raw_output)
    gen_stage, labels = classify_generation(
        output_profile=output_profile,
        raw_output=raw_output,
        parsed=None,
        extracted_code=code,
        extraction_errors=errors,
        schema_valid=True,
        structure_valid=bool(code),
    )
    predict = code or ""
    return {
        "parsed": None,
        "predict": predict,
        "test_code": None,
        "extraction_errors": errors,
        "schema_valid": True,
        "gen_stage": gen_stage,
        "gen_labels": labels,
        "record": {"instruction": instruction, "predict": predict},
    }


def run_condition_local(
    *,
    model: dict[str, Any],
    condition: dict[str, Any],
    experiment: dict[str, Any],
    results_root: Path,
    model_cache: Path,
    tritonbench_repo: Path,
    parser_binary: Path | None = None,
) -> dict[str, Any]:
    experiment_id = experiment["experiment_id"]
    dataset = experiment.get("benchmark", {}).get("dataset", "simp")
    limit = effective_benchmark_limit(
        model, experiment, experiment.get("_hf_defaults", {})
    )

    condition["grammar_version"] = experiment.get("grammar_version", "1.0")
    condition["prompt_template_version"] = experiment.get(
        "prompt_template_version", "tritonbench_v1"
    )

    writer = ArtifactWriter(results_root, experiment_id)
    run_id = new_run_id()
    model = dict(model)
    model["model_revision"] = get_model_revision(model["hf_id"])

    generator = VllmGenerator(model, str(model_cache))
    items = load_alpaca_dataset(tritonbench_repo, dataset)[:limit]

    predictions: list[dict] = []
    attempt_rows: list[dict] = []
    cond_id = condition_id(model["name"], condition)
    pred_path = (
        writer.experiment_dir / "predictions" / f"{cond_id}.jsonl"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    repair_budget = int(condition.get("repair_budget", 0))

    for task_idx, item in enumerate(items):
        repair_error: str | None = None
        final_proc: dict[str, Any] | None = None
        final_gen: dict[str, Any] | None = None
        final_messages: list[dict[str, str]] | None = None

        for repair_round in range(repair_budget + 1):
            messages = build_messages(
                item,
                output_profile=condition["output_profile"],
                prompt_variant=condition["prompt_variant"],
                xgrammar_enabled=condition.get("xgrammar_enabled", False),
                repair_error=repair_error,
            )
            gen = generator.generate(
                messages,
                temperature=condition["temperature"],
                top_p=condition["top_p"],
                max_tokens=condition["max_output_tokens"],
                seed=condition["seed"] + repair_round,
                output_profile=condition["output_profile"],
                xgrammar_enabled=condition.get("xgrammar_enabled", False),
            )
            proc = process_generation_output(
                output_profile=condition["output_profile"],
                raw_output=gen["text"],
                instruction=item["instruction"],
            )
            final_proc = proc
            final_gen = gen
            final_messages = messages
            if repair_round >= repair_budget:
                break
            if proc["gen_stage"] == "ok" and proc.get("predict"):
                break
            repair_error = "; ".join(proc.get("extraction_errors", [])) or gen["text"][:2000]

        assert final_proc is not None and final_gen is not None and final_messages is not None
        proc = final_proc
        gen = final_gen
        messages = final_messages

        predictions.append(proc["record"])

        structure_valid = False
        if (
            condition["output_profile"] == "minitriton_json"
            and proc["predict"]
            and parser_binary
        ):
            check = run_mini_triton_parser(parser_binary, proc["predict"])
            structure_valid = check["valid"]
            proc["gen_stage"] = "ok" if structure_valid else "structure"
        elif condition["output_profile"] == "tritonbench_python":
            structure_valid = bool(proc["predict"])

        base_dir = writer.attempt_dir(
            model["name"],
            model.get("quantization", "unknown"),
            condition["output_profile"],
            condition["decoding_mode"],
            task_idx,
            condition["seed"],
            repair_round,
        )

        summary = build_summary(
            output_profile=condition["output_profile"],
            gen_stage=proc["gen_stage"],
            gen_labels=proc["gen_labels"],
            eval_result=None,
        )
        if condition["output_profile"] == "minitriton_json":
            summary["schema_valid"] = proc["schema_valid"]
            summary["structure_valid"] = structure_valid

        metadata = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "model_name": model["name"],
            "hf_model_id": model["hf_id"],
            "condition": condition,
            "task_idx": task_idx,
            "task_id": f"tritonbench_t_{task_idx:03d}",
            "repair_round": repair_round,
        }

        artifact_paths = writer.save_attempt(
            base_dir=base_dir,
            metadata=metadata,
            metrics={},
            failure={"failure_labels": summary["failure_labels"]}
            if summary["failure_labels"]
            else None,
            prompt=json.dumps(messages, indent=2),
            raw_output=gen["text"],
            parsed_output=proc.get("parsed"),
            extracted_code=proc["predict"] or None,
            test_code=proc.get("test_code"),
            environment=collect_environment(
                modal_function="run_generation_condition",
            ),
        )

        metrics = build_attempt_metrics(
            run_id=run_id,
            experiment_id=experiment_id,
            model=model,
            condition=condition,
            task_id=metadata["task_id"],
            task_idx=task_idx,
            seed=condition["seed"],
            attempt=repair_round,
            repair_round=repair_round,
            artifact_paths=artifact_paths,
            gen_stats=gen,
            eval_stats={},
            summary=summary,
        )
        writer.write_json(base_dir / "metrics.json", metrics)
        attempt_rows.append(metrics)

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")

    eval_summary: dict[str, Any] = {}
    if condition["output_profile"] == "tritonbench_python":
        runner = TritonBenchRunner(tritonbench_repo)
        eval_out_dir = writer.experiment_dir / "eval" / model["name"] / cond_id
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        eval_summary = runner.evaluate_predictions(
            pred_path,
            eval_out_dir,
            run_perf=experiment.get("mode") != "pilot",
        )
        writer.write_json(eval_out_dir / "eval_summary.json", eval_summary)

        total = max(eval_summary.get("total_predictions", 1), 1)
        call_rate = eval_summary["phase1_call_acc"]["passed"] / total
        exe_rate = eval_summary["phase2_exec_acc"]["passed"] / total
        speedup = eval_summary.get("phase3_efficiency", {}).get(
            "speedup_vs_pytorch"
        )
        min_speedup = experiment.get("performance", {}).get(
            "min_speedup_vs_pytorch", 0.0
        )
        perf_ok = speedup is None or speedup >= min_speedup

        for task_idx in range(len(items)):
            base_dir = writer.attempt_dir(
                model["name"],
                model.get("quantization", "unknown"),
                condition["output_profile"],
                condition["decoding_mode"],
                task_idx,
                condition["seed"],
                0,
            )
            metrics_path = base_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            has_code = bool(m.get("code_extracted"))
            op_passed = has_code and call_rate > 0
            exe_passed = has_code and exe_rate > 0
            summary = build_summary(
                output_profile=condition["output_profile"],
                gen_stage="ok" if has_code else "extract",
                gen_labels=[],
                eval_result={
                    "exe_acc_passed": exe_passed,
                    "perf_passed": perf_ok,
                },
                op_passed=op_passed,
            )
            m.update(
                {
                    "compiled": summary["compiled"],
                    "correctness_passed": summary["correctness_passed"],
                    "performance_passed": summary["performance_passed"],
                    "overall_passed": summary["overall_passed"],
                    "failure_stage": summary["failure_stage"],
                    "failure_labels": summary["failure_labels"],
                    "speedup_vs_pytorch": speedup,
                    "compile_duration_ms": eval_summary["phase1_call_acc"].get(
                        "duration_ms", 0
                    ),
                }
            )
            writer.write_json(metrics_path, m)

    return {
        "condition_id": cond_id,
        "predictions_path": str(pred_path),
        "eval_summary": eval_summary,
        "attempt_count": len(attempt_rows),
    }
