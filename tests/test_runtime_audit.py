from __future__ import annotations

from scripts.audit_submission import imported_roots


def test_runtime_import_audit_accepts_base_modules() -> None:
    roots = imported_roots("import torch\nimport numpy as np\nfrom pathlib import Path", "ok.py")
    assert roots == {"numpy", "pathlib", "torch"}


def test_runtime_import_audit_detects_training_dependency() -> None:
    roots = imported_roots("from safetensors.torch import load_file", "bad.py")
    assert roots == {"safetensors"}
