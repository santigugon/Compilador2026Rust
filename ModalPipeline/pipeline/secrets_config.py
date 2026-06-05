"""Modal secret names and Hugging Face env vars used by the pipeline."""

from __future__ import annotations

import os

# Modal Secret object name (create via `modal secret create <this-name> ...`)
DEFAULT_MODAL_HF_SECRET_NAME = "hf-token"

# Env var names injected from that Modal secret into containers
HF_TOKEN_ENV = "HF_TOKEN"
HUGGINGFACE_HUB_TOKEN_ENV = "HUGGING_FACE_HUB_TOKEN"


def modal_hf_secret_name() -> str:
    """Override locally with MODAL_HF_SECRET_NAME if you use a different secret name."""
    return os.environ.get("MODAL_HF_SECRET_NAME", DEFAULT_MODAL_HF_SECRET_NAME)


def hf_token_from_env() -> str | None:
    return os.environ.get(HF_TOKEN_ENV) or os.environ.get(HUGGINGFACE_HUB_TOKEN_ENV)
