"""Run oracle-backed multi-step Agent tasks against one recorded server manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from ledger_x.app.agent import ReconciliationAgent
from ledger_x.app.client import LocalModel
from ledger_x.app.data import DEFAULT_DB
from ledger_x.app.tools import ToolService
from ledger_x.paths import RUNTIME_DIR


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    a, b = int(pos), int(pos) if pos.is_integer() else int(pos) + 1
    return ordered[a] * (b - pos) + ordered[b] * (pos - a) if a != b else ordered[a]


def normalize(value):
    return "".join(str(value).replace(",", "").split()).lower()


def score(case, trace):
    calls = [tool["call"]["function"]["name"]
             for round_item in trace.get("rounds", []) for tool in round_item.get("tools", [])]
    tool_chain = all(name in calls for name in case["required_tools"])
    evidence = normalize(json.dumps(trace.get("evidence", []), ensure_ascii=False))
    answer = normalize(trace.get("answer") or "")
    evidence_match = all(normalize(fact["value"]) in evidence for fact in case["facts"])
    answer_match = all(normalize(fact["value"]) in answer for fact in case["facts"])
    return {"tool_chain_match": tool_chain, "evidence_match": evidence_match,
            "answer_fact_match": answer_match,
            "task_success": trace.get("status") == "answered" and tool_chain and evidence_match and answer_match}


def summarize(records, wall_seconds):
    traces = [record["trace"] for record in records if "trace" in record]
    elapsed = [trace["elapsed_ms"] for trace in traces]
    llm = [trace["timing"]["llm_ms"] for trace in traces]
    tools = [trace["timing"]["tool_ms"] for trace in traces]
    total = len(records)
    return {
        "tasks": total,
        "errors": sum("error" in record for record in records),
        "task_success": sum(record.get("score", {}).get("task_success", False) for record in records),
        "task_success_rate": sum(record.get("score", {}).get("task_success", False) for record in records) / total if total else None,
        "tool_chain_accuracy": sum(record.get("score", {}).get("tool_chain_match", False) for record in records) / total if total else None,
        "evidence_accuracy": sum(record.get("score", {}).get("evidence_match", False) for record in records) / total if total else None,
        "answer_fact_accuracy": sum(record.get("score", {}).get("answer_fact_match", False) for record in records) / total if total else None,
        "agent_p50_ms": percentile(elapsed, .5), "agent_p95_ms": percentile(elapsed, .95),
        "llm_p50_ms": percentile(llm, .5), "llm_p95_ms": percentile(llm, .95),
        "tool_p50_ms": percentile(tools, .5), "wall_seconds": wall_seconds,
        "mean_llm_rounds": statistics.mean(trace["timing"]["llm_rounds"] for trace in traces) if traces else None,
        "mean_tool_calls": statistics.mean(trace["timing"]["tool_calls"] for trace in traces) if traces else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default=str(RUNTIME_DIR / "agent-workload.jsonl"))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["warm", "tune", "test"], default="test")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    model = LocalModel()
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["served_model"] != model.config.model:
        parser.error("Manifest and LEDGERX_MODEL mismatch")
    data = [json.loads(line) for line in Path(args.workload).read_text().splitlines()]
    selected = [item for item in data if item["split"] == args.split][:args.limit]
    service = ToolService(args.db, [f"M{i:03d}" for i in range(1, 6)])
    agent = ReconciliationAgent(service, model=model)
    records, start = [], time.perf_counter()
    with (out / "records.jsonl").open("w") as log:
        for repeat in range(args.repeats):
            for case in selected:
                record = {"case_id": case["id"], "repeat": repeat}
                try:
                    trace = agent.run(case["question"])
                    record.update(trace=trace, score=score(case, trace))
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
    (out / "config.json").write_text(json.dumps({**vars(args), "server": manifest}, ensure_ascii=False, indent=2))
    summary = summarize(records, time.perf_counter() - start)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
