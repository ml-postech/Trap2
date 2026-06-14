"""Resolve the root directory that holds released adapters.

Defaults to the repository's ``released_updates/`` directory; override with the
``TRAP2_ADAPTER_ROOT`` environment variable to point at your own adapters.
Adapters are referenced by clean names, e.g. ``b32_lora_gtsrb`` (unprotected)
and ``b32_lora_gtsrb_trap2`` (Trap2-protected).
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_ROOT = os.environ.get(
    "TRAP2_ADAPTER_ROOT", os.path.join(_REPO_ROOT, "released_updates")
)
