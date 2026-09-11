"""Project paths for the local workspace."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_WORKLOAD = ARTIFACTS_DIR / "qwen3-14b-workload-v2.jsonl"
PROMOTION_REPORT = ARTIFACTS_DIR / "evidence" / "latest-benchmark.json"
GATE_REPORT = RUNTIME_DIR / "gate" / "result.json"
