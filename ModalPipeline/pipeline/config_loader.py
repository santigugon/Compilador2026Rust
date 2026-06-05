from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_hf_models(path: Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIGS_DIR / "hf_models.yaml")


def load_experiment(path: Path) -> dict[str, Any]:
    return load_yaml(path)


def resolve_models(hf_config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = hf_config.get("defaults", {})
    models = []
    for entry in hf_config.get("models", []):
        if not entry.get("enabled", defaults.get("enabled", True)):
            continue
        merged = _deep_merge(defaults, entry)
        merged.setdefault("base_model", defaults.get("base_model"))
        merged["_explicit_model_keys"] = sorted(entry.keys())
        models.append(merged)
    return models


def effective_benchmark_limit(
    model: dict[str, Any],
    experiment: dict[str, Any],
    hf_defaults: dict[str, Any],
) -> int:
    explicit_model_keys = set(model.get("_explicit_model_keys", []))
    if "benchmark_limit" in explicit_model_keys and model.get("benchmark_limit") is not None:
        return int(model["benchmark_limit"])

    experiment_limit = experiment.get("benchmark", {}).get("limit")
    if experiment_limit is not None:
        return int(experiment_limit)

    for source in (model.get("benchmark_limit"), hf_defaults.get("benchmark_limit")):
        if source is not None:
            return int(source)
    return 166


def expand_matrix(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = experiment.get("matrix", {})
    mode = experiment.get("mode", "pilot")

    profiles = matrix.get("output_profiles", ["tritonbench_python"])
    decoding = matrix.get("decoding_modes", ["xgrammar_off"])
    sampling = matrix.get("sampling", [{"temperature": 0.0, "top_p": 1.0}])
    max_tokens_list = matrix.get("max_output_tokens", [2048])
    prompts = matrix.get("prompt_variants", ["minimal"])
    repairs = matrix.get("repair_budgets", [0])
    runtimes = matrix.get("runtime_limits", [{"name": "default"}])
    seeds = experiment.get("seeds", [0])

    if mode == "full":
        profiles = matrix.get("output_profiles", profiles)
        decoding = matrix.get("decoding_modes", decoding)
        sampling = matrix.get(
            "sampling",
            [
                {"temperature": 0.0, "top_p": 1.0},
                {"temperature": 0.2, "top_p": 0.95},
                {"temperature": 0.7, "top_p": 0.8},
            ],
        )
        max_tokens_list = matrix.get("max_output_tokens", [512, 1024, 2048, 4096])
        prompts = matrix.get(
            "prompt_variants",
            ["minimal", "full_rules", "one_shot", "two_shot", "repair_compile"],
        )
        repairs = matrix.get("repair_budgets", [0, 1, 2])
        seeds = experiment.get("seeds", [0, 1, 2])

    conditions: list[dict[str, Any]] = []
    for profile in profiles:
        for dec in decoding:
            for sample in sampling:
                for max_tok in max_tokens_list:
                    for prompt_var in prompts:
                        for repair in repairs:
                            for runtime in runtimes:
                                for seed in seeds:
                                    conditions.append(
                                        {
                                            "output_profile": profile,
                                            "decoding_mode": dec,
                                            "xgrammar_enabled": dec == "xgrammar_on",
                                            "temperature": sample.get("temperature", 0.0),
                                            "top_p": sample.get("top_p", 1.0),
                                            "max_output_tokens": max_tok,
                                            "prompt_variant": prompt_var,
                                            "repair_budget": repair,
                                            "runtime_limit": runtime,
                                            "seed": seed,
                                        }
                                    )
    return conditions


def condition_id(model_name: str, cond: dict[str, Any]) -> str:
    parts = [
        model_name,
        cond["output_profile"],
        cond["decoding_mode"],
        f"t{cond['temperature']}",
        f"p{cond['top_p']}",
        f"tok{cond['max_output_tokens']}",
        cond["prompt_variant"],
        f"repair{cond['repair_budget']}",
        cond.get("runtime_limit", {}).get("name", "default"),
        f"seed{cond['seed']}",
    ]
    return "_".join(str(p) for p in parts)
