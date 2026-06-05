#!/usr/bin/env python3
"""Sync HF_TOKEN from .env to Modal secret hf-token (or MODAL_HF_SECRET_NAME)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.env import load_project_dotenv, sync_modal_hf_secret_from_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the Modal secret",
    )
    args = p.parse_args()

    if not load_project_dotenv():
        print(f"No .env found at {ROOT / '.env'} — copy .env.example to .env first.")
        sys.exit(1)

    name = sync_modal_hf_secret_from_env(force=args.force)
    print(f"Modal secret '{name}' is ready (HF_TOKEN from .env).")


if __name__ == "__main__":
    main()
