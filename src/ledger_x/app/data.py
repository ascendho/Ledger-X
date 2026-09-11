"""Deterministic synthetic ledger. Never imports or connects to real finances."""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import random

import duckdb
from ledger_x.paths import RUNTIME_DIR

VERSION = "synthetic-cny-v1"
AS_OF = "2026-09-15"
CHANNELS = ("alipay", "wechat", "card")
DEFAULT_DB = RUNTIME_DIR / "ledger.duckdb"

DDL = """
CREATE TABLE metadata (key VARCHAR PRIMARY KEY, value VARCHAR NOT NULL);
CREATE TABLE merchants (merchant_id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL);
CREATE TABLE fee_rules (rule_id VARCHAR PRIMARY KEY, channel VARCHAR, valid_from DATE,
                       valid_to DATE, rate_bps BIGINT, refund_fee_returned BOOLEAN);
CREATE TABLE payments (transaction_id VARCHAR PRIMARY KEY, merchant_id VARCHAR,
                       channel VARCHAR, paid_date DATE, amount_fen BIGINT);
CREATE TABLE refunds (refund_id VARCHAR PRIMARY KEY, transaction_id VARCHAR,
                      refund_date DATE, amount_fen BIGINT);
CREATE TABLE settlement_batches (batch_id VARCHAR PRIMARY KEY, merchant_id VARCHAR,
                       channel VARCHAR, settlement_date DATE, due_date DATE, currency VARCHAR);
CREATE TABLE settlement_items (item_id VARCHAR PRIMARY KEY, batch_id VARCHAR,
                       kind VARCHAR, source_id VARCHAR, amount_fen BIGINT, charged_fee_fen BIGINT);
CREATE TABLE bank_receipts (receipt_id VARCHAR PRIMARY KEY, batch_id VARCHAR,
                       received_date DATE, amount_fen BIGINT);
CREATE VIEW expected_items AS
WITH unique_items AS (
 SELECT DISTINCT batch_id, kind, source_id FROM settlement_items
), payment_items AS (
 SELECT u.batch_id, u.kind, u.source_id, p.amount_fen,
        (p.amount_fen*r.rate_bps + 5000)//10000 AS expected_fee_fen
 FROM unique_items u JOIN payments p ON u.source_id=p.transaction_id AND u.kind='payment'
 JOIN fee_rules r ON r.channel=p.channel AND p.paid_date>=r.valid_from AND p.paid_date<r.valid_to
), refund_items AS (
 SELECT u.batch_id, u.kind, u.source_id, -r.amount_fen AS amount_fen, 0 AS expected_fee_fen
 FROM unique_items u JOIN refunds r ON u.source_id=r.refund_id AND u.kind='refund'
)
SELECT * FROM payment_items UNION ALL SELECT * FROM refund_items;
CREATE VIEW expected_batches AS
SELECT b.*, coalesce(e.expected_fen,0)::BIGINT AS expected_fen,
       coalesce(c.claimed_fen,0)::BIGINT AS claimed_fen,
       coalesce(c.fee_fen,0)::BIGINT AS charged_fee_fen,
       coalesce(e.fee_fen,0)::BIGINT AS expected_fee_fen,
       coalesce(c.item_count,0)-coalesce(e.item_count,0) AS duplicate_count
FROM settlement_batches b
LEFT JOIN (SELECT batch_id, sum(amount_fen-expected_fee_fen) AS expected_fen,
           sum(expected_fee_fen) AS fee_fen, count(*) AS item_count
           FROM expected_items GROUP BY batch_id) e USING(batch_id)
LEFT JOIN (SELECT batch_id, sum(amount_fen-charged_fee_fen) AS claimed_fen,
           sum(charged_fee_fen) AS fee_fen, count(*) AS item_count
           FROM settlement_items GROUP BY batch_id) c USING(batch_id);
"""


def generate(path=DEFAULT_DB, count=100_000, seed=42):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; use a new dataset path")
    if count < 90:
        raise ValueError("At least 90 payments required to populate every batch")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    batches, rules = [], []
    for month in range(3, 9):
        settlement = date(2026, month + 1, 1) - timedelta(days=1)
        for merchant in range(1, 6):
            for channel in CHANNELS:
                bid = f"B{month:02d}-M{merchant:03d}-{channel}"
                batches.append((bid, f"M{merchant:03d}", channel, settlement,
                                settlement + timedelta(days=2), "CNY"))
    for channel, offset in zip(CHANNELS, (0, 5, 10)):
        rules += [(f"R-{channel}-v1", channel, "2026-03-01", "2026-06-01", 55 + offset, False),
                  (f"R-{channel}-v2", channel, "2026-06-01", "2027-01-01", 60 + offset, False)]
    payments, refunds, items = [], [], []
    # Independent generation oracle: this is not computed with application SQL.
    gold = {b[0]: {"expected_fen": 0, "claimed_fen": 0, "injected": []} for b in batches}
    for i in range(count):
        batch_index = i % len(batches)
        bid, mid, channel, settlement, _, _ = batches[batch_index]
        day = date(2026, settlement.month, rng.randint(1, 28))
        amount = rng.randint(500, 200_000)
        tid = f"T{i + 1:07d}"
        rate = (55 if day.month < 6 else 60) + CHANNELS.index(channel) * 5
        fee = (amount * rate + 5000) // 10000
        charged_fee = fee + (20 if batch_index % 13 == 3 else 0)
        payments.append((tid, mid, channel, day, amount))
        items.append((f"I{len(items):08d}", bid, "payment", tid, amount, charged_fee))
        gold[bid]["expected_fen"] += amount - fee
        gold[bid]["claimed_fen"] += amount - charged_fee
        if rng.random() < 0.07:
            rday = day + timedelta(days=rng.randint(1, 20))
            refund_amount = rng.randint(1, amount)
            rid = f"RF{i + 1:07d}"
            refunds.append((rid, tid, rday, refund_amount))
            # A refund belongs to its own later settlement month, never retroactively
            # deducted from the payment's month. September refunds remain unsettled.
            rbid = f"B{rday.month:02d}-{mid}-{channel}"
            if rbid in gold:
                items.append((f"I{len(items):08d}", rbid, "refund", rid, -refund_amount, 0))
                gold[rbid]["expected_fen"] -= refund_amount
                gold[rbid]["claimed_fen"] -= refund_amount
    first_payment = {}
    for item in items:
        if item[2] == "payment":
            first_payment.setdefault(item[1], item)
    receipts = []
    for i, batch in enumerate(batches):
        bid, _, _, _, due, _ = batch
        anomaly = i % 13
        if anomaly == 2:
            old = first_payment[bid]
            items.append((f"I{len(items):08d}", *old[1:]))
            gold[bid]["claimed_fen"] += old[4] - old[5]
            gold[bid]["injected"].append("duplicate_record")
        if anomaly == 3:
            gold[bid]["injected"].append("fee_deviation")
        actual = gold[bid]["claimed_fen"]
        received = due
        if anomaly == 0:
            gold[bid]["injected"].append("missing_receipt")
            actual = 0
        else:
            if anomaly == 1:
                received += timedelta(days=8)
                gold[bid]["injected"].append("delayed_receipt")
            if anomaly == 4:
                actual -= 10_000
                gold[bid]["injected"].append("short_payment")
            receipts.append((f"DEP-{bid}", bid, received, actual))
        gold[bid]["actual_fen"] = actual
        gold[bid]["difference_fen"] = actual - gold[bid]["expected_fen"]
    con = duckdb.connect(str(path))
    try:
        con.execute("BEGIN")
        con.execute(DDL)
        metadata = {"version": VERSION, "seed": str(seed), "payments": str(count),
                    "as_of": AS_OF, "timezone": "Asia/Shanghai", "currency": "CNY"}
        for table, rows in [("metadata", list(metadata.items())),
                            ("merchants", [(f"M{i:03d}", f"示例商户{i}") for i in range(1, 6)]),
                            ("fee_rules", rules), ("payments", payments), ("refunds", refunds),
                            ("settlement_batches", batches), ("settlement_items", items),
                            ("bank_receipts", receipts)]:
            con.executemany(f"INSERT INTO {table} VALUES ({','.join('?' for _ in rows[0])})", rows)
        con.execute("COMMIT")
    finally:
        con.close()
    oracle = path.with_suffix(".oracle.json")
    oracle.write_text(json.dumps({"metadata": metadata, "batches": gold}, ensure_ascii=False, indent=2))
    return {"database": str(path), "oracle": str(oracle), **metadata}
