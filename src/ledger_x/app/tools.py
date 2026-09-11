"""Typed read-only tools, parameterized SQL, trusted merchant scope."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import time
from typing import Literal

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ledger_x.app.data import AS_OF


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MerchantQuery(Args):
    query: str = Field(min_length=1, max_length=100)


class Period(Args):
    merchant_id: str = Field(pattern=r"^M\d{3}$")
    start_date: date = Field(description="结算日期下界，包含；YYYY-MM-DD")
    end_date: date = Field(description="结算日期上界，不包含；YYYY-MM-DD")
    as_of: date = Field(default=date.fromisoformat(AS_OF), description="到账观察截止日期，包含")
    currency: Literal["CNY"] = "CNY"

    @model_validator(mode="after")
    def interval(self):
        if self.start_date >= self.end_date or (self.end_date - self.start_date).days > 366:
            raise ValueError("Date interval must be positive and at most 366 days")
        return self


class Summary(Period):
    group_by: Literal["channel", "day"] = "channel"


class Exceptions(Period):
    exception_type: Literal["all", "missing_receipt", "delayed_receipt", "duplicate_record",
                            "fee_deviation", "short_payment"] = "all"
    limit: int = Field(default=20, ge=1, le=100)


class Batch(Args):
    batch_id: str = Field(pattern=r"^B\d{2}-M\d{3}-(alipay|wechat|card)$")
    as_of: date = date.fromisoformat(AS_OF)


class Transaction(Args):
    transaction_id: str = Field(pattern=r"^T\d{7}$")


class Rule(Args):
    merchant_id: str = Field(pattern=r"^M\d{3}$")
    channel: Literal["alipay", "wechat", "card"]
    as_of: date = Field(description="交易日期，用于匹配当时生效的规则")


REGISTRY = {
    "resolve_merchant": (MerchantQuery, "解析商户名称或编号；返回候选，歧义时向用户澄清。"),
    "query_settlement_summary": (Summary, "按结算日期区间汇总应结算、已到账、差额；金额单位分。"),
    "list_reconciliation_exceptions": (Exceptions, "列出结算区间内异常批次，按差额绝对值降序，支持类型筛选。"),
    "get_settlement_batch": (Batch, "查看结算批次、汇总、前20条明细和到账证据。"),
    "get_transaction_detail": (Transaction, "查看支付交易、退款及所属结算批次。"),
    "get_settlement_rule": (Rule, "按交易日期查询生效费率；退款不返还支付手续费。"),
    "reconcile_settlement_batch": (Batch, "确定性复算批次差额，给出规则与异常证据。"),
}


def tool_schemas():
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": model.model_json_schema()}}
            for name, (model, description) in REGISTRY.items()]


RECONCILE = """
SELECT e.*, coalesce(r.actual_fen,0)::BIGINT AS actual_fen,
       (coalesce(r.actual_fen,0)-e.expected_fen)::BIGINT AS difference_fen,
       coalesce(r.receipt_count,0) AS receipt_count, r.latest_received_date,
       coalesce(r.actual_fen,0)-e.claimed_fen AS payout_difference_fen
FROM expected_batches e LEFT JOIN (
 SELECT batch_id, sum(amount_fen) AS actual_fen, count(*) AS receipt_count,
        max(received_date) AS latest_received_date
 FROM bank_receipts WHERE received_date<=? GROUP BY batch_id
) r USING(batch_id)
"""


def json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def classify(row, as_of):
    result = []
    if row["receipt_count"] == 0 and row["due_date"] <= as_of:
        result.append("missing_receipt")
    if row["latest_received_date"] and row["latest_received_date"] > row["due_date"]:
        result.append("delayed_receipt")
    if row["duplicate_count"] > 0:
        result.append("duplicate_record")
    # Duplicate fees are explained by duplicated rows; fee deviation is identified
    # separately by comparing the unique recorded payment fees to rule-derived fees.
    if row.get("unique_fee_deviation_fen", 0) != 0:
        result.append("fee_deviation")
    if row["receipt_count"] > 0 and row["payout_difference_fen"] < 0:
        result.append("short_payment")
    return result


class ToolService:
    def __init__(self, db: str | Path, allowed_merchants=("M001",)):
        self.db = str(db)
        self.allowed = frozenset(allowed_merchants)
        if not self.allowed:
            raise ValueError("Empty merchant scope")

    @staticmethod
    def rows(con, sql, args=()):
        cursor = con.execute(sql, args)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _scope(self, merchant_id):
        if merchant_id not in self.allowed:
            raise PermissionError("Merchant is outside this session's authorized scope")

    def _reconcile(self, con, batch_id, as_of):
        records = self.rows(con, RECONCILE + " WHERE e.batch_id=?", [as_of, batch_id])
        if not records:
            raise LookupError("Batch not found")
        row = records[0]
        self._scope(row["merchant_id"])
        fees = self.rows(con, """SELECT coalesce(sum(c.fee-e.expected_fee_fen),0)::BIGINT AS deviation
          FROM (SELECT source_id, max(charged_fee_fen) AS fee FROM settlement_items
                WHERE batch_id=? AND kind='payment' GROUP BY source_id) c
          JOIN expected_items e ON e.source_id=c.source_id AND e.batch_id=? AND e.kind='payment'""",
          [batch_id, batch_id])
        row["unique_fee_deviation_fen"] = fees[0]["deviation"]
        row["exceptions"] = classify(row, as_of)
        row["evidence"] = {"batch_id": batch_id, "rules": self.rows(con, """
          SELECT DISTINCT r.rule_id, r.rate_bps, r.valid_from, r.valid_to
          FROM settlement_items i JOIN payments p ON i.source_id=p.transaction_id AND i.kind='payment'
          JOIN fee_rules r ON p.channel=r.channel AND p.paid_date>=r.valid_from AND p.paid_date<r.valid_to
          WHERE i.batch_id=?""", [batch_id]), "receipts": self.rows(con,
          "SELECT * FROM bank_receipts WHERE batch_id=? AND received_date<=?", [batch_id, as_of])}
        return row

    def execute(self, name, arguments):
        start = time.perf_counter()
        if name not in REGISTRY:
            raise ValueError("Unknown tool")
        args = REGISTRY[name][0].model_validate(arguments)
        if hasattr(args, "merchant_id"):
            self._scope(args.merchant_id)
        with duckdb.connect(self.db, read_only=True) as con:
            metadata = dict(con.execute("SELECT key,value FROM metadata").fetchall())
            if name == "resolve_merchant":
                candidates = self.rows(con, "SELECT * FROM merchants WHERE contains(name,?) OR merchant_id=?",
                                       [args.query, args.query])
                result = {"candidates": [r for r in candidates if r["merchant_id"] in self.allowed]}
            elif name == "query_settlement_summary":
                group = "channel" if args.group_by == "channel" else "settlement_date"
                # Identifier comes solely from the enum above, never raw model text.
                result = {"rows": self.rows(con, f"""WITH r AS ({RECONCILE})
                  SELECT {group} AS dimension, sum(expected_fen)::BIGINT AS expected_fen,
                         sum(actual_fen)::BIGINT AS actual_fen,
                         sum(difference_fen)::BIGINT AS difference_fen, count(*) AS batch_count
                  FROM r WHERE merchant_id=? AND settlement_date>=? AND settlement_date<?
                  GROUP BY {group} ORDER BY {group}""",
                  [args.as_of, args.merchant_id, args.start_date, args.end_date])}
            elif name == "list_reconciliation_exceptions":
                batches = self.rows(con, """SELECT batch_id FROM settlement_batches
                  WHERE merchant_id=? AND settlement_date>=? AND settlement_date<? ORDER BY batch_id""",
                  [args.merchant_id, args.start_date, args.end_date])
                records = [self._reconcile(con, b["batch_id"], args.as_of) for b in batches]
                records = [r for r in records if r["exceptions"] and
                           (args.exception_type == "all" or args.exception_type in r["exceptions"])]
                records.sort(key=lambda r: (-abs(r["difference_fen"]), r["batch_id"]))
                result = {"total": len(records), "rows": records[:args.limit],
                          "truncated": len(records) > args.limit}
            elif name in {"get_settlement_batch", "reconcile_settlement_batch"}:
                result = self._reconcile(con, args.batch_id, args.as_of)
                if name == "get_settlement_batch":
                    result["items_preview"] = self.rows(con,
                        "SELECT * FROM settlement_items WHERE batch_id=? ORDER BY item_id LIMIT 20", [args.batch_id])
                    result["items_total"] = con.execute("SELECT count(*) FROM settlement_items WHERE batch_id=?",
                                                       [args.batch_id]).fetchone()[0]
            elif name == "get_transaction_detail":
                payments = self.rows(con, "SELECT * FROM payments WHERE transaction_id=?", [args.transaction_id])
                if not payments:
                    raise LookupError("Transaction not found")
                self._scope(payments[0]["merchant_id"])
                result = {"payment": payments[0], "refunds": self.rows(con,
                    "SELECT * FROM refunds WHERE transaction_id=? ORDER BY refund_date", [args.transaction_id]),
                    "settlements": self.rows(con, """SELECT * FROM settlement_items
                      WHERE (kind='payment' AND source_id=?) OR (kind='refund' AND source_id IN
                        (SELECT refund_id FROM refunds WHERE transaction_id=?)) ORDER BY batch_id,item_id""",
                      [args.transaction_id, args.transaction_id])}
            else:
                result = {"rules": self.rows(con, """SELECT * FROM fee_rules
                  WHERE channel=? AND valid_from<=? AND valid_to>?""", [args.channel, args.as_of, args.as_of])}
        return json_safe({"tool": name, "arguments": args.model_dump(mode="json"), "result": result,
                          "provenance": {"dataset": metadata, "sql_template": name + ":v1",
                                         "money_unit": "fen", "difference": "actual_minus_expected",
                                         "time_basis": "settlement_date", "end_date_exclusive": True},
                          "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)})
