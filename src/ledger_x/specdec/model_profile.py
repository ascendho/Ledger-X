"""Single source of truth for the target model and service shape."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    repository: str = "Qwen/Qwen3-14B-FP8"
    path: str = "models/Qwen3-14B-FP8"
    served_name: str = "qwen3-14b-fp8-ledger-x"
    host: str = "0.0.0.0"
    port: int = 2011
    gpus: tuple[int, ...] = (2,)
    tensor_parallel: int = 1
    max_model_len: int = 32768
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 4096
    vllm_version: str = "0.23.0"

    @classmethod
    def from_env(cls) -> "ModelProfile":
        """Allow explicit test/deployment overrides while preserving one default."""
        default = cls()
        gpus = tuple(int(item) for item in os.getenv(
            "LEDGERX_GPUS", ",".join(map(str, default.gpus))).split(",") if item)
        profile = cls(
            repository=os.getenv("LEDGERX_MODEL_REPOSITORY", default.repository),
            path=os.getenv("LEDGERX_MODEL_PATH", default.path),
            served_name=os.getenv("LEDGERX_MODEL", default.served_name),
            host=os.getenv("LEDGERX_HOST", default.host),
            port=int(os.getenv("LEDGERX_PORT", default.port)),
            gpus=gpus,
            tensor_parallel=int(os.getenv("LEDGERX_TENSOR_PARALLEL", default.tensor_parallel)),
            max_model_len=int(os.getenv("LEDGERX_MAX_MODEL_LEN", default.max_model_len)),
            max_num_seqs=int(os.getenv("LEDGERX_MAX_NUM_SEQS", default.max_num_seqs)),
            max_num_batched_tokens=int(os.getenv(
                "LEDGERX_MAX_NUM_BATCHED_TOKENS", default.max_num_batched_tokens)),
            vllm_version=os.getenv("LEDGERX_VLLM_VERSION", default.vllm_version),
        )
        if not profile.gpus or profile.tensor_parallel != len(profile.gpus):
            raise ValueError("LEDGERX_TENSOR_PARALLEL must equal the number of LEDGERX_GPUS")
        if profile.port == 2010:
            raise ValueError("Ledger-X refuses to serve on the former production port 2010")
        return profile

    def manifest_fields(self) -> dict:
        result = asdict(self)
        result["gpus"] = list(self.gpus)
        download = Path(self.path) / ".ledgerx-download.json"
        if download.exists():
            try:
                result["download"] = json.loads(download.read_text())
            except (OSError, ValueError):
                result["download"] = {"status": "unreadable"}
        return result


PROFILE = ModelProfile.from_env()
