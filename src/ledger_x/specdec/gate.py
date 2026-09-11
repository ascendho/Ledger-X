"""Destructive-to-dev-only compatibility gate, always restores Qwen3-14B AR.

Run on the company server with --replace-dev. The exact API-server PID is
validated before SIGTERM; production PIDs and shared process groups are untouched.
Results are evidence, not a performance benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from ledger_x.paths import PROJECT_ROOT, RUNTIME_DIR
from ledger_x.specdec.model_profile import PROFILE

ROOT = PROJECT_ROOT
RUN = RUNTIME_DIR / "gate"
BASE = f"http://127.0.0.1:{PROFILE.port}"
DEV_INTERPRETER_PREFIX = os.getenv("LEDGERX_DEV_INTERPRETER_PREFIX", "")
DEV_PYTHON = os.getenv("LEDGERX_DEV_PYTHON", sys.executable)


def status(message):
    """Logging failure must never prevent service cleanup or restoration."""
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


def request(path, body=None, timeout=120):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def dev_pid():
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"vllm.entrypoints.openai.api_server" not in args:
            continue
        if b"--port" in args and args[args.index(b"--port") + 1] == str(PROFILE.port).encode():
            if DEV_INTERPRETER_PREFIX and not args[0].decode(errors="ignore").startswith(DEV_INTERPRETER_PREFIX):
                raise RuntimeError("Unexpected development interpreter; refusing to stop")
            matches.append(int(entry.name))
    if len(matches) > 1:
        raise RuntimeError("Ambiguous development PID")
    return matches[0] if matches else None


def stop_dev():
    pid = dev_pid()
    if pid is None:
        return
    status(f"Stopping validated dev API PID {pid}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(120):
        if dev_pid() is None:
            # The API process can disappear before TP workers have released CUDA
            # contexts and shared-memory queues. A wider grace period prevents the
            # following server (especially the restore path) racing those workers.
            time.sleep(12)
            return
        time.sleep(1)
    raise RuntimeError("Dev server did not stop gracefully; refusing force kill")


def wait_ready(process, seconds=900):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server startup failed: exit {process.returncode}")
        try:
            if request("/v1/models", timeout=3)["data"][0]["id"] == PROFILE.served_name:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("Dev startup timed out")


def probes():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROFILE.path, local_files_only=True)
    tool = {"type": "function", "function": {"name": "query_settlement_summary",
            "description": "查询商户结算汇总", "parameters": {"type": "object", "properties": {
                "merchant_id": {"type": "string"}, "start_date": {"type": "string"},
                "end_date": {"type": "string"}, "group_by": {"type": "string", "enum": ["channel", "day"]}},
                "required": ["merchant_id", "start_date", "end_date", "group_by"]}}}
    cases = ["查询 M001 2026-08-01 到 2026-09-01 的结算，按渠道汇总。",
             "查询 M002 2026-07-01 到 2026-08-01 的结算，按天汇总。",
             "只回答：你好。", "背景资料：" + "这是合成对账数据，不是真实账目。" * 450 +
             "\n查询 M003 2026-06-01 到 2026-07-01 的结算，按渠道汇总。"]
    bodies = []
    for case in cases:
        scope = json.dumps({"tenant": "gate-synthetic", "merchants": ["M001", "M002", "M003"],
                            "memory_write": True}, separators=(",", ":"))
        prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": "Synthetic gate only.\nLEDGERX_SCOPE_JSON:" + scope},
                    {"role": "user", "content": case}], tools=[tool],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not isinstance(prompt, list):
            prompt = prompt["input_ids"]
        bodies.append({"model": PROFILE.served_name, "prompt": prompt,
                       "temperature": 0, "seed": 42, "max_tokens": 160, "return_token_ids": True})
    return bodies


def run_cases(bodies, concurrency=4, repeat=2):
    def one(body):
        r = request("/v1/completions", body)
        choice = r["choices"][0]
        if not choice.get("token_ids"):
            raise RuntimeError("Token IDs missing: cannot verify exactness")
        return {"token_ids": choice["token_ids"], "text": choice["text"],
                "finish_reason": choice["finish_reason"], "usage": r["usage"]}
    serial = [one(body) for body in bodies]
    # Repeat after prefix caching, reorder and interleave different-length requests.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        repeated = list(pool.map(one, list(reversed(bodies)) * repeat))
    return {"serial": serial, "interleaved": repeated}


def compare_section(baseline, candidate):
    return len(baseline) == len(candidate) and all(
        a["token_ids"] == b["token_ids"] for a, b in zip(baseline, candidate))


def diff_summary(report, repeat):
    reference = report["modes"].get("ar", {})
    expected_interleaved = list(reversed(reference.get("serial", []))) * repeat
    summary = {"tool_call": {"matches": 0, "total": 0}, "non_tool": {"matches": 0, "total": 0},
               "first_difference": None}
    for mode, data in report["modes"].items():
        checks = (("serial", reference.get("serial", []), data.get("serial", [])),
                  ("interleaved", expected_interleaved, data.get("interleaved", [])))
        for section, baseline, candidate in checks:
            for index, (left, right) in enumerate(zip(baseline, candidate)):
                key = "tool_call" if left.get("text", "").startswith("<tool_call>") or right.get("text", "").startswith("<tool_call>") else "non_tool"
                summary[key]["total"] += 1
                ok = left.get("token_ids") == right.get("token_ids")
                summary[key]["matches"] += int(ok)
                if not ok and summary["first_difference"] is None:
                    summary["first_difference"] = {"mode": mode, "section": section, "index": index,
                                                   "ar_text": left.get("text"),
                                                   "candidate_text": right.get("text"),
                                                   "class": key}
    return summary


def launch_args(mode, args):
    cmd = [sys.executable, "-m", "ledger_x.specdec.launch", mode, "--budget", str(args.budget), "--exec"]
    cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    if args.disable_prefix_cache:
        cmd.append("--disable-prefix-cache")
    if args.async_scheduling:
        cmd.append("--async-scheduling")
    return cmd


def file_hash(name):
    return hashlib.sha256((ROOT / "src/ledger_x/specdec" / name).read_bytes()).hexdigest()


def modes_for(profile):
    if profile == "ar-determinism":
        return ("ar",)
    if profile == "diagnostic":
        return ("ar", "empty", "wrong")
    return ("ar", "schema", "retrieval", "full")


def activity_ok(profile, activity):
    if profile != "full-proposer":
        return True
    # `ngram` uses vLLM's built-in proposer and therefore does not write a
    # project stats artifact. Its output equivalence is still checked above;
    # source activity is proven by the three project proposer modes.
    return all(activity.get(mode, {}).get("counters", {}).get("proposed_tokens", 0) > 0
               for mode in ("schema", "retrieval", "full"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--replace-dev", action="store_true")
    parser.add_argument("--profile", choices=["diagnostic", "full-proposer", "ar-determinism"],
                        default="diagnostic")
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--budget", type=int, choices=[2, 4, 8, 16])
    args = parser.parse_args()
    if args.budget is None:
        args.budget = 16 if args.profile == "full-proposer" else 2
    if not args.replace_dev:
        parser.error("Explicit --replace-dev required; temporarily restarts only port 2011")
    if DEV_PYTHON and Path(sys.executable).resolve() != Path(DEV_PYTHON).resolve():
        raise RuntimeError("Gate must run with isolated server interpreter")
    RUN.mkdir(parents=True, exist_ok=True)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    report = {"kind": args.profile, "passed": False, "diagnostic_passed": False,
              "full_proposer_passed": False, "modes": {}, "activity": {}, "restored": False,
              "settings": {"concurrency": args.concurrency, "repeat": args.repeat,
                           "prefix_cache": not args.disable_prefix_cache,
                           "async_scheduling": args.async_scheduling,
                           "max_num_seqs": args.max_num_seqs, "budget": args.budget},
              "model": PROFILE.manifest_fields(),
              "artifacts": {"proposer_sha256": file_hash("proposer.py"),
                            "drafts_sha256": file_hash("drafts.py"),
                            "bridge_sha256": file_hash("bridge.py"),
                            "vllm": importlib.metadata.version("vllm")}}
    original_present = dev_pid() is not None
    if not original_present:
        raise RuntimeError("Expected original development service to be running")
    process = None
    try:
        bodies = probes()
        modes = modes_for(args.profile)
        for mode in modes:
            stop_dev()
            status(f"Starting gate mode={mode}")
            stats = RUN / f"{mode}-stats.json"
            stats.unlink(missing_ok=True)
            env = os.environ.copy()
            env["LEDGERX_GATE_PROFILE"] = args.profile
            env["LEDGERX_STATS_PATH"] = str(stats)
            with (RUN / f"{mode}.log").open("w") as log:
                process = subprocess.Popen(launch_args(mode, args), cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, env=env)
            wait_ready(process)
            report["modes"][mode] = run_cases(bodies, args.concurrency, args.repeat)
            if stats.exists():
                report["activity"][mode] = json.loads(stats.read_text())
            (RUN / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            status(f"Completed gate mode={mode}")
        reference = report["modes"]["ar"]
        expected_repeat = list(reversed(reference["serial"])) * args.repeat
        report["comparisons"] = {
            mode: compare_section(reference["serial"], data["serial"])
            and compare_section(expected_repeat, data["interleaved"])
            for mode, data in report["modes"].items()}
        report["analysis"] = diff_summary(report, args.repeat)
        equivalent = all(report["comparisons"].values())
        active = activity_ok(args.profile, report["activity"])
        report["mode_passed"] = {
            mode: bool(report["comparisons"].get(mode))
            and report["activity"].get(mode, {}).get("counters", {}).get("proposed_tokens", 0) > 0
            for mode in ("schema", "retrieval", "full") if mode in report["comparisons"]
        }
        report["activity_passed"] = active
        report["passed"] = equivalent and active
        report["diagnostic_passed"] = args.profile == "diagnostic" and report["passed"]
        report["full_proposer_passed"] = args.profile == "full-proposer" and report["passed"]
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        status(report["error"])
    finally:
        try:
            stop_dev()
            with (RUN / "restore.log").open("a") as log:
                restored = subprocess.Popen(
                    [sys.executable, "-m", "ledger_x.specdec.launch", "ar", "--max-num-seqs", "1", "--exec"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_ready(restored)
            report["restored"] = True
            status("Qwen3-14B autoregressive service restored")
        except BaseException as exc:
            report["restore_error"] = f"{type(exc).__name__}: {exc}"
        (RUN / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    status(json.dumps({k: v for k, v in report.items() if k != "modes"}, ensure_ascii=False))
    return 0 if report["passed"] and report["restored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
