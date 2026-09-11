"""Agent session state, context management, and evidence-only memory."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import time

from ledger_x.app.data import AS_OF


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


@dataclass
class AgentSession:
    tenant: str
    merchants: tuple[str, ...]
    as_of: str = AS_OF
    case_id: str = ""
    max_rounds: int = 8
    started_at: float = field(default_factory=time.time)

    def scope_json(self) -> str:
        return json.dumps({"tenant": self.tenant, "merchants": list(self.merchants)},
                          separators=(",", ":"))

    def key(self) -> str:
        return self.tenant + ":" + ",".join(self.merchants)


@dataclass(frozen=True)
class MemoryFact:
    """A verified fact extracted from tool output, never from model prose."""

    tool: str
    name: str
    value: object
    scope: dict
    confidence: str = "evidence_only"
    created_at_round: int = 0
    salience: int = 1

    def compact(self):
        return {"tool": self.tool, "name": self.name, "value": self.value,
                "scope": self.scope, "confidence": self.confidence,
                "round": self.created_at_round, "salience": self.salience}


class MemoryPolicy:
    """Write policy for memory. Only verified tool evidence is durable."""

    ALLOWED = {
        "batch_id", "merchant_id", "channel", "settlement_date", "due_date",
        "expected_fen", "actual_fen", "claimed_fen", "difference_fen",
        "receipt_count", "latest_received_date", "payout_difference_fen",
        "transaction_id", "refund_id", "rule_id", "rate_bps", "exception",
    }
    HIGH_SALIENCE = {"batch_id", "difference_fen", "exception", "rule_id",
                     "transaction_id", "refund_id"}

    def allow(self, name, value):
        return name in self.ALLOWED and value not in (None, "", [], {})

    def salience(self, name):
        return 3 if name in self.HIGH_SALIENCE else 1


class EvidenceNotebook:
    """Working memory for the current case. It stores tool facts, not model prose."""

    IMPORTANT = {
        "batch_id", "merchant_id", "channel", "settlement_date", "due_date",
        "expected_fen", "actual_fen", "claimed_fen", "difference_fen",
        "receipt_count", "latest_received_date", "payout_difference_fen",
        "transaction_id", "refund_id", "rule_id", "rate_bps",
    }

    def __init__(self, previous=None, *, session=None, memory_policy=None):
        self.session = session
        self.memory_policy = memory_policy or MemoryPolicy()
        self.facts = []
        for item in previous or []:
            if isinstance(item, MemoryFact):
                self.facts.append(item)
            else:
                self.facts.append(MemoryFact(
                    tool=item.get("tool", "memory"),
                    name=item.get("name", ""),
                    value=item.get("value"),
                    scope=item.get("scope", {}),
                    confidence=item.get("confidence", "evidence_only"),
                    created_at_round=item.get("round", item.get("created_at_round", 0)),
                    salience=item.get("salience", 1),
                ))

    def add_tool_result(self, payload, *, round_index=0):
        before = len(self.facts)
        tool = payload.get("tool", "unknown")
        arguments = payload.get("arguments", {})
        result = payload.get("result", {})
        for key, value in _walk({"arguments": arguments, "result": result}):
            if key in self.IMPORTANT and self.memory_policy.allow(key, value):
                self._add(tool, key, value, arguments, round_index)
        exceptions = []
        for key, value in _walk(result):
            if key == "exceptions" and isinstance(value, list):
                exceptions.extend(str(item) for item in value)
        for item in exceptions:
            self._add(tool, "exception", item, arguments, round_index)
        return len(self.facts) - before

    def _scope(self, arguments):
        scope = {"tenant": self.session.tenant if self.session else "synthetic-demo"}
        merchants = self.session.merchants if self.session else ()
        if arguments.get("merchant_id"):
            scope["merchant_id"] = arguments["merchant_id"]
        elif arguments.get("batch_id"):
            match = re.match(r"^B\d{2}-(M\d{3})-", str(arguments["batch_id"]))
            if match:
                scope["merchant_id"] = match.group(1)
        elif len(merchants) == 1:
            scope["merchant_id"] = merchants[0]
        return scope

    def _add(self, tool, name, value, arguments, round_index):
        item = MemoryFact(tool=tool, name=name, value=value, scope=self._scope(arguments),
                          created_at_round=round_index,
                          salience=self.memory_policy.salience(name))
        if item not in self.facts:
            self.facts.append(item)

    def summary(self, limit=24):
        return self.compact_summary(limit)

    def compact_summary(self, limit=24):
        if not self.facts:
            return "无已验证工具证据。"
        lines = []
        ranked = sorted(enumerate(self.facts), key=lambda item: (item[1].salience, item[0]))
        for _, fact in ranked[-limit:]:
            scope = ",".join(f"{k}={v}" for k, v in sorted(fact.scope.items()) if v)
            lines.append(f"- [{fact.tool}] {fact.name}={fact.value} ({scope})")
        return "\n".join(lines)

    def values_for_checking(self):
        values = set()
        for fact in self.facts:
            value = str(fact.value)
            if len(value) >= 3:
                values.add(value)
            if re.fullmatch(r"-?\d+", value):
                values.add(value.replace(",", ""))
        return values

    def snapshot(self):
        return [fact.compact() for fact in self.facts]


class CaseMemory:
    """Tenant/scope-isolated long-term memory backed by verified evidence."""

    def __init__(self):
        self._facts = {}

    def read(self, session, question, limit=16):
        facts = list(self._facts.get(session.key(), []))
        if not facts:
            return []
        tokens = set(re.findall(r"B\d{2}-M\d{3}-(?:alipay|wechat|card)|T\d{7}|RF\d{7}|R-[\w-]+|M\d{3}|missing_receipt|delayed_receipt|duplicate_record|fee_deviation|short_payment", question))
        if tokens:
            facts = [f for f in facts if str(f.value) in tokens or f.scope.get("merchant_id") in tokens
                     or f.name in tokens]
        ranked = sorted(enumerate(facts), key=lambda item: (item[1].salience, item[0]))
        return [fact for _, fact in ranked[-limit:]]

    def write(self, session, facts):
        current = list(self._facts.get(session.key(), []))
        written = []
        for fact in facts:
            if fact.scope.get("merchant_id") and fact.scope["merchant_id"] not in session.merchants:
                continue
            if fact not in current:
                current.append(fact)
                written.append(fact)
        self._facts[session.key()] = current[-200:]
        return written

    def snapshot(self, session):
        return [fact.compact() for fact in self._facts.get(session.key(), [])]


class ContextManager:
    """Builds bounded prompts from short-term context and evidence memory."""

    DATE_RE = re.compile(r"20\d{2}[-年]\d{1,2}[-月]\d{1,2}|20\d{2}[-年]")
    MERCHANT_RE = re.compile(r"M\d{3}|商户")
    DIRECT_OBJECT_RE = re.compile(r"B\d{2}-M\d{3}-(?:alipay|wechat|card)|T\d{7}|RF\d{7}|规则|费率")

    def __init__(self, *, max_context_chars=2400, max_memory_facts=16):
        self.max_context_chars = max_context_chars
        self.max_memory_facts = max_memory_facts

    def clarify(self, question, session, memory_reads):
        filled = []
        questions = []
        has_merchant = bool(self.MERCHANT_RE.search(question) or self.DIRECT_OBJECT_RE.search(question))
        if not has_merchant and len(session.merchants) == 1:
            filled.append({"field": "merchant_id", "value": session.merchants[0],
                           "source": "authorized_scope"})
            has_merchant = True
        if not has_merchant:
            questions.append("请指定要查询的商户编号，例如 M001。")
        needs_period = not self.DIRECT_OBJECT_RE.search(question)
        has_date = bool(self.DATE_RE.search(question))
        if needs_period and not has_date:
            questions.append("请给出结算日期范围，例如 2026-03-01 到 2026-09-01。")
        return questions, filled

    def build(self, *, session, playbook, notebook, memory_reads):
        memory_lines = []
        for fact in memory_reads[-self.max_memory_facts:]:
            scope = ",".join(f"{k}={v}" for k, v in sorted(fact.scope.items()) if v)
            memory_lines.append(f"- [{fact.tool}] {fact.name}={fact.value} ({scope})")
        blocks = {
            "playbook": playbook,
            "working_memory": notebook.compact_summary(),
            "retrieved_memory": "\n".join(memory_lines) if memory_lines else "无可注入历史证据。",
        }
        raw = json.dumps(blocks, ensure_ascii=False)
        compacted = raw
        if len(compacted) > self.max_context_chars:
            compacted = compacted[:self.max_context_chars] + "\n...[context truncated]"
        return {
            "message": "LEDGERX_AGENT_CONTEXT:\n" + compacted,
            "blocks": blocks,
            "budget": {"raw_chars": len(raw), "sent_chars": len(compacted),
                       "memory_facts": len(memory_reads), "working_facts": len(notebook.facts)},
        }
