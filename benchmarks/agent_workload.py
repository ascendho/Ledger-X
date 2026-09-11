"""Create deterministic, oracle-backed multi-step Ledger-X Agent tasks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import duckdb

from ledger_x.app.data import DEFAULT_DB
from ledger_x.paths import RUNTIME_DIR


def _fact(name, value):
    return {"name": name, "value": value}


def cases(db=DEFAULT_DB, seed=42):
    db = Path(db)
    oracle = json.loads(db.with_suffix(".oracle.json").read_text())
    result = []

    # Root-cause playbooks: list candidates, then recompute the selected batch.
    for merchant in (f"M{i:03d}" for i in range(1, 6)):
        candidates = [(bid, row) for bid, row in oracle["batches"].items()
                      if f"-{merchant}-" in bid and row["injected"]]
        candidates.sort(key=lambda item: abs(item[1]["difference_fen"]), reverse=True)
        for rank, (batch_id, row) in enumerate(candidates[:6]):
            result.append({
                "category": "exception_triage",
                "question": f"调查 {merchant} 2026-03-01 到 2026-09-01 的结算异常，定位差额绝对值第{rank + 1}大的异常批次并核验原因。",
                "required_tools": ["list_reconciliation_exceptions", "reconcile_settlement_batch"],
                "facts": [_fact("batch_id", batch_id), _fact("difference_fen", row["difference_fen"]),
                          _fact("exception", row["injected"][0])],
            })

    # Direct batch audit playbooks use the independent accounting oracle.
    batches = sorted(oracle["batches"].items())
    for batch_id, row in batches[:30]:
        result.append({
            "category": "batch_audit",
            "question": f"重新核算批次 {batch_id}，报告应结算、实际到账、差额和异常原因。",
            "required_tools": ["reconcile_settlement_batch"],
            "facts": [_fact("batch_id", batch_id), _fact("expected_fen", row["expected_fen"]),
                      _fact("actual_fen", row["actual_fen"]), _fact("difference_fen", row["difference_fen"])],
        })

    with duckdb.connect(str(db), read_only=True) as con:
        transactions = con.execute("""SELECT transaction_id, merchant_id, channel, amount_fen
            FROM payments ORDER BY transaction_id LIMIT 30""").fetchall()
        rules = con.execute("""SELECT rule_id, channel, rate_bps, valid_from
            FROM fee_rules ORDER BY rule_id, valid_from LIMIT 30""").fetchall()

    for transaction_id, merchant_id, channel, amount_fen in transactions:
        result.append({
            "category": "transaction_trace",
            "question": f"追踪交易 {transaction_id} 的商户、渠道、原始金额、退款和所属结算批次。",
            "required_tools": ["get_transaction_detail"],
            "facts": [_fact("transaction_id", transaction_id), _fact("merchant_id", merchant_id),
                      _fact("channel", channel), _fact("amount_fen", amount_fen)],
        })

    for rule_id, channel, rate_bps, valid_from in rules:
        for merchant_id in (f"M{i:03d}" for i in range(1, 6)):
            result.append({
                "category": "fee_rule",
                "question": f"核验 {merchant_id} 在 {valid_from} 这一天通过 {channel} 渠道适用的手续费规则和费率，并给出规则编号。",
                "required_tools": ["get_settlement_rule"],
                "facts": [_fact("rule_id", rule_id), _fact("merchant_id", merchant_id),
                          _fact("channel", channel), _fact("rate_bps", rate_bps)],
            })

    random.Random(seed).shuffle(result)
    assert len(result) == 120
    for index, item in enumerate(result):
        item.update(id=f"agent-{index:03d}", split="warm" if index < 20 else "tune" if index < 40 else "test")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=str(RUNTIME_DIR / "agent-workload.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases(args.db, args.seed)))
    print(json.dumps({"output": str(output), "cases": 120, "warm": 20, "tune": 20, "test": 80}, ensure_ascii=False))


if __name__ == "__main__":
    main()
