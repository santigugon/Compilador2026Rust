from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = PACKAGE_ROOT / "prompts"


def load_prompt_template(name: str) -> str:
    mapping = {
        "minimal": "tritonbench_v1.txt",
        "full_rules": "minitriton_full_rules_v1.txt",
        "one_shot": "tritonbench_v1.txt",
        "two_shot": "tritonbench_two_shot_v1.txt",
        "repair_compile": "repair_compile_v1.txt",
    }
    filename = mapping.get(name, "tritonbench_v1.txt")
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def build_messages(
    item: dict,
    *,
    output_profile: str,
    prompt_variant: str,
    xgrammar_enabled: bool = False,
    repair_error: str | None = None,
) -> list[dict[str, str]]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"

    if output_profile == "minitriton_json":
        system = load_prompt_template(
            "full_rules" if prompt_variant in ("full_rules", "minimal") else prompt_variant
        )
    else:
        system = load_prompt_template(
            "minimal" if prompt_variant == "minimal" else prompt_variant
        )
        if xgrammar_enabled:
            system += (
                "\n\nStructured output mode is enabled. Return a single JSON object only, "
                "with exactly these fields: instruction and predict. The predict field "
                "must contain the complete Python module source as a string. Do not use "
                "markdown fences or prose outside the JSON object."
            )

    if repair_error:
        system += "\n\n" + load_prompt_template("repair_compile").format(
            error_text=repair_error
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
