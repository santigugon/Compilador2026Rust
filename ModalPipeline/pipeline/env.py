from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pipeline.secrets_config import (
    HF_TOKEN_ENV,
    HUGGINGFACE_HUB_TOKEN_ENV,
    hf_token_from_env,
    modal_hf_secret_name,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PACKAGE_ROOT / ".env"


def load_project_dotenv() -> bool:
    """Load ModalPipeline/.env into os.environ (no-op if missing)."""
    if not DOTENV_PATH.exists():
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(DOTENV_PATH, override=False)
        return True
    except ImportError:
        # Minimal parser fallback when python-dotenv is not installed
        for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return True


def require_hf_token() -> str:
    load_project_dotenv()
    token = hf_token_from_env()
    if not token:
        raise RuntimeError(
            f"Missing {HF_TOKEN_ENV} in environment. "
            f"Copy .env.example to .env and set your Hugging Face token."
        )
    return token


def sync_modal_hf_secret_from_env(*, force: bool = False) -> str:
    """
    Push HF_TOKEN from .env into the named Modal secret (default: hf-token).
    Containers read it via modal.Secret.from_name(...) in app.py.
    """
    token = require_hf_token()
    name = modal_hf_secret_name()

    cmd = [
        "modal",
        "secret",
        "create",
        *(["--force"] if force else []),
        name,
        f"{HF_TOKEN_ENV}={token}",
        f"{HUGGINGFACE_HUB_TOKEN_ENV}={token}",
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if "already exists" in err.lower() or "duplicate" in err.lower():
            proc = subprocess.run(
                [
                    "modal",
                    "secret",
                    "create",
                    "--force",
                    name,
                    f"{HF_TOKEN_ENV}={token}",
                    f"{HUGGINGFACE_HUB_TOKEN_ENV}={token}",
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                return name
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Failed to update Modal secret '{name}': {err}")
        raise RuntimeError(
            f"modal secret create failed for '{name}': {err}\n"
            f"Run: python scripts/sync_secrets_from_env.py --force"
        )

    return name
