from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
GRAMMAR_DIR = PACKAGE_ROOT / "grammar"


def load_schema(name: str) -> dict:
    path = GRAMMAR_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def schema_for_profile(output_profile: str) -> dict:
    if output_profile == "minitriton_json":
        return load_schema("minitriton_response.schema.json")
    return load_schema("tritonbench_envelope.schema.json")


def build_guided_decoding_params(
    output_profile: str,
    xgrammar_enabled: bool,
) -> dict | None:
    if not xgrammar_enabled:
        return None
    schema = schema_for_profile(output_profile)
    return {
        "guided_json": schema,
    }
