"""Print or exec the single-model server command. Never targets 2010."""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path
import hashlib
from datetime import datetime, timezone

from ledger_x.paths import GATE_REPORT, PROJECT_ROOT, PROMOTION_REPORT, RUNTIME_DIR
from ledger_x.specdec.model_profile import PROFILE

ROOT = PROJECT_ROOT
GATE = GATE_REPORT
PROMOTION = PROMOTION_REPORT


CUSTOM_MODES = {"schema", "retrieval", "full"}
DIAGNOSTIC_MODES = {"empty", "wrong", "ngram"}
MODES = {"ar"} | CUSTOM_MODES | DIAGNOSTIC_MODES


def _profile_matches(data):
    return (data.get("repository") == PROFILE.repository
            and data.get("path") == PROFILE.path
            and data.get("served_name") == PROFILE.served_name
            and data.get("gpus") == list(PROFILE.gpus)
            and data.get("tensor_parallel") == PROFILE.tensor_parallel)


def gate_allows(mode, budget, max_num_seqs, async_scheduling, prefix_cache):
    if not GATE.exists():
        return False
    result = json.loads(GATE.read_text())
    settings = result.get("settings", {})
    artifacts = result.get("artifacts", {})
    counters = result.get("activity", {}).get(mode, {}).get("counters", {})
    derived_mode_pass = (result.get("comparisons", {}).get(mode) is True
                         and counters.get("proposed_tokens", 0) > 0)
    mode_pass = (result.get("full_proposer_passed") is True if mode == "full"
                 else result.get("mode_passed", {}).get(mode, derived_mode_pass) is True)
    return (mode_pass
            and result.get("restored") is True
            and artifacts.get("vllm") == PROFILE.vllm_version
            and _profile_matches(result.get("model", {}))
            and artifacts.get("proposer_sha256") == hashlib.sha256(Path(__file__).with_name("proposer.py").read_bytes()).hexdigest()
            and artifacts.get("drafts_sha256") == hashlib.sha256(Path(__file__).with_name("drafts.py").read_bytes()).hexdigest()
            and artifacts.get("bridge_sha256") == hashlib.sha256(Path(__file__).with_name("bridge.py").read_bytes()).hexdigest()
            and settings.get("max_num_seqs") == max_num_seqs
            and settings.get("async_scheduling") == async_scheduling
            and settings.get("prefix_cache") == prefix_cache
            and int(settings.get("budget", 0)) >= budget)


def benchmark_allows(mode, budget, max_num_seqs, async_scheduling, prefix_cache):
    """Require workload evidence before an accelerated mode is operational.

    The component gate is deliberately small and is useful for catching bridge
    crashes and obvious token drift.  It is not a deployment-quality sample.
    A promoted run is therefore bound to the exact server/code configuration
    recorded by ``experiments.report``.
    """
    if not PROMOTION.exists():
        return False
    report = json.loads(PROMOTION.read_text())
    server = report.get("candidate_server", {})
    return (report.get("criteria_passed") is True
            and report.get("candidate") == mode
            and server.get("mode") == mode
            and server.get("vllm") == PROFILE.vllm_version
            and _profile_matches(server.get("profile", {}))
            and server.get("proposer_sha256") == hashlib.sha256(Path(__file__).with_name("proposer.py").read_bytes()).hexdigest()
            and server.get("drafts_sha256") == hashlib.sha256(Path(__file__).with_name("drafts.py").read_bytes()).hexdigest()
            and server.get("bridge_sha256") == hashlib.sha256(Path(__file__).with_name("bridge.py").read_bytes()).hexdigest()
            and server.get("max_num_seqs") == max_num_seqs
            and server.get("async_scheduling") == async_scheduling
            and server.get("prefix_cache") == prefix_cache
            and int(server.get("budget", 0)) >= budget)


def command(mode, budget=8, async_scheduling=False, prefix_cache=True,
            max_num_seqs=PROFILE.max_num_seqs):
    if mode not in MODES:
        raise ValueError(mode)
    if max_num_seqs < 1:
        raise ValueError("max_num_seqs must be positive")
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", PROFILE.path,
            "--served-model-name", PROFILE.served_name, "--host", PROFILE.host,
            "--port", str(PROFILE.port), "--tensor-parallel-size", str(PROFILE.tensor_parallel),
            "--max-model-len", str(PROFILE.max_model_len),
            "--gpu-memory-utilization", "0.90", "--max-num-seqs", str(max_num_seqs),
            "--max-num-batched-tokens", str(PROFILE.max_num_batched_tokens),
            "--reasoning-parser", "qwen3",
            "--tool-call-parser", "hermes", "--enable-auto-tool-choice",
            "--language-model-only", "--generation-config", "vllm"]
    if prefix_cache:
        args += ["--enable-prefix-caching"]
    if not async_scheduling:
        args += ["--no-async-scheduling"]
    if mode != "ar":
        if mode == "ngram":
            spec = {"method": "ngram", "num_speculative_tokens": budget,
                    "prompt_lookup_min": 5, "prompt_lookup_max": 7}
        else:
            spec = {"method": "custom_class", "model": "ledger_x.specdec.proposer.LedgerXProposer",
                    "num_speculative_tokens": budget}
        args += ["--speculative-config", json.dumps(spec)]
    return args


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--budget", type=int, choices=[2, 4, 8, 16], default=8)
    parser.add_argument("--operational", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=PROFILE.max_num_seqs)
    parser.add_argument("--exec", dest="execute", action="store_true")
    args = parser.parse_args()
    cmd = command(args.mode, args.budget, args.async_scheduling,
                  not args.disable_prefix_cache, args.max_num_seqs)
    if not args.execute:
        print("CUDA_VISIBLE_DEVICES=" + ",".join(map(str, PROFILE.gpus))
              + " LEDGERX_DRAFT_MODE=" + args.mode + " " + shlex.join(cmd))
        return
    import importlib.metadata
    if importlib.metadata.version("vllm") != PROFILE.vllm_version:
        raise RuntimeError(f"Only inspected vLLM {PROFILE.vllm_version} is supported")
    if not Path(PROFILE.path, ".ledgerx-download.json").exists():
        raise RuntimeError("Model download manifest missing; refusing an unverified checkpoint")
    if args.mode in CUSTOM_MODES:
        gate_profile = os.environ.get("LEDGERX_GATE_PROFILE")
        verification = (gate_profile == "full-proposer"
                        and args.max_num_seqs == 1 and not args.async_scheduling
                        and not args.disable_prefix_cache)
        experiment_flag = os.environ.get("LEDGERX_EXPERIMENT")
        experiment = (experiment_flag == "1"
                      and args.max_num_seqs == 1 and not args.async_scheduling
                      and not args.disable_prefix_cache
                      and gate_allows(args.mode, args.budget, args.max_num_seqs,
                                      args.async_scheduling,
                                      not args.disable_prefix_cache))
        promoted = benchmark_allows(args.mode, args.budget, args.max_num_seqs,
                                    args.async_scheduling,
                                    not args.disable_prefix_cache)
        if not verification and not experiment and not promoted:
            raise RuntimeError("No matching promoted workload benchmark: accelerated serving is blocked")
    manifest = {"mode": args.mode, "budget": args.budget,
                "served_model": PROFILE.served_name, "vllm": PROFILE.vllm_version,
                "profile": PROFILE.manifest_fields(),
                "gpus": list(PROFILE.gpus), "tensor_parallel": PROFILE.tensor_parallel,
                "max_num_seqs": args.max_num_seqs,
                "async_scheduling": args.async_scheduling,
                "mtp_enabled": False,
                "prefix_cache": not args.disable_prefix_cache,
                "started_utc": datetime.now(timezone.utc).isoformat(), "args": cmd,
                "proposer_sha256": hashlib.sha256(Path(__file__).with_name("proposer.py").read_bytes()).hexdigest(),
                "drafts_sha256": hashlib.sha256(Path(__file__).with_name("drafts.py").read_bytes()).hexdigest(),
                "bridge_sha256": hashlib.sha256(Path(__file__).with_name("bridge.py").read_bytes()).hexdigest()}
    path = RUNTIME_DIR / "service-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, PROFILE.gpus))
    os.environ["LEDGERX_DRAFT_MODE"] = args.mode
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
