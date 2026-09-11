"""Compare paired end-to-end Agent benchmark runs.

This report is intentionally separate from the one-shot function-call benchmark:
that module
measures one-shot function-call generation, while this one measures the whole
multi-round Agent loop including LLM calls, tool execution, policy checks,
memory updates, and final answer generation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from ledger_x.paths import ARTIFACTS_DIR


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    a, b = math.floor(pos), math.ceil(pos)
    return ordered[a] * (b - pos) + ordered[b] * (pos - a) if a != b else ordered[a]


def load_run(path):
    path = Path(path)
    return {
        "config": json.loads((path / "config.json").read_text()),
        "summary": json.loads((path / "summary.json").read_text()),
        "records": [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()],
    }


def trace_map(run):
    return {
        (row["case_id"], row["repeat"]): row["trace"]
        for row in run["records"]
        if "trace" in row and row.get("score", {}).get("task_success") is True
    }


def score_summary(run):
    records = run["records"]
    traces = [row["trace"] for row in records if "trace" in row]
    elapsed = [trace["elapsed_ms"] for trace in traces]
    llm = [trace["timing"]["llm_ms"] for trace in traces]
    tools = [trace["timing"]["tool_ms"] for trace in traces]
    rounds = [trace["timing"]["llm_rounds"] for trace in traces]
    calls = [trace["timing"]["tool_calls"] for trace in traces]
    total = len(records)
    success = sum(row.get("score", {}).get("task_success", False) for row in records)
    return {
        "tasks": total,
        "successful_tasks": success,
        "task_success_rate": success / total if total else None,
        "errors": sum("error" in row for row in records),
        "agent_p50_ms": percentile(elapsed, .5),
        "agent_p95_ms": percentile(elapsed, .95),
        "llm_p50_ms": percentile(llm, .5),
        "llm_p95_ms": percentile(llm, .95),
        "tool_p50_ms": percentile(tools, .5),
        "mean_llm_rounds": statistics.mean(rounds) if rounds else None,
        "mean_tool_calls": statistics.mean(calls) if calls else None,
    }


def paired_reduction(baseline, candidate, samples=2000, seed=42):
    left, right = trace_map(baseline), trace_map(candidate)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("Runs have no paired successful Agent tasks")

    def reduction(selected, field):
        a = statistics.median(left[key][field] for key in selected)
        b = statistics.median(right[key][field] for key in selected)
        return 1 - b / a if a else None

    def timing_reduction(selected, field):
        a = statistics.median(left[key]["timing"][field] for key in selected)
        b = statistics.median(right[key]["timing"][field] for key in selected)
        return 1 - b / a if a else None

    rng = random.Random(seed)
    values = []
    llm_values = []
    for _ in range(samples):
        selected = [rng.choice(keys) for _ in keys]
        values.append(reduction(selected, "elapsed_ms"))
        llm_values.append(timing_reduction(selected, "llm_ms"))
    values.sort()
    llm_values.sort()
    return {
        "paired_successful_tasks": len(keys),
        "agent_median_reduction": reduction(keys, "elapsed_ms"),
        "agent_reduction_ci95": [values[int(.025 * samples)], values[int(.975 * samples) - 1]],
        "llm_median_reduction": timing_reduction(keys, "llm_ms"),
        "llm_reduction_ci95": [llm_values[int(.025 * samples)], llm_values[int(.975 * samples) - 1]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ar", required=True, help="Directory created by benchmarks.agent_benchmark for AR mode")
    parser.add_argument("--candidate", required=True, help="Directory created by benchmarks.agent_benchmark for accelerated mode")
    parser.add_argument("--candidate-name", default="full")
    parser.add_argument("--out", default=str(ARTIFACTS_DIR / "evidence" / "agent-e2e-comparison.json"))
    args = parser.parse_args()

    ar = load_run(args.ar)
    candidate = load_run(args.candidate)
    reduction = paired_reduction(ar, candidate)
    ar_summary = score_summary(ar)
    candidate_summary = score_summary(candidate)
    same_workload = (
        ar["config"].get("workload") == candidate["config"].get("workload")
        and ar["config"].get("split") == candidate["config"].get("split")
        and ar["config"].get("limit") == candidate["config"].get("limit")
        and ar["config"].get("repeats") == candidate["config"].get("repeats")
    )
    success_preserved = candidate_summary["task_success_rate"] >= ar_summary["task_success_rate"]
    speedup_verified = (
        same_workload
        and success_preserved
        and reduction["agent_median_reduction"] is not None
        and reduction["agent_median_reduction"] > 0
        and reduction["agent_reduction_ci95"][0] > 0
    )
    report = {
        "claim": (
            f"{args.candidate_name} improves end-to-end Agent latency"
            if speedup_verified else
            "No verified end-to-end Agent speedup"
        ),
        "criteria_passed": speedup_verified,
        "same_workload": same_workload,
        "success_preserved": success_preserved,
        "baseline": ar_summary,
        "candidate": candidate_summary,
        "paired_reduction": reduction,
        "measurement_note": (
            "End-to-end latency covers the complete Agent run: policy checks, context/memory handling, "
            "all LLM rounds, tool execution, and final answer generation. It is expected to be smaller "
            "than the isolated function-call-generation speedup when the task has multiple LLM rounds."
        ),
        "baseline_server": ar["config"].get("server", {}),
        "candidate_server": candidate["config"].get("server", {}),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
