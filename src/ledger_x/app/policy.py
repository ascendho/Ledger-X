"""Policy guardrails for the read-only agent."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re

from ledger_x.app.tools import REGISTRY


WRITE_INTENT = re.compile(
    r"(转账|打款|付款|修改|删除|更新|写入|执行退款|发起退款|办理退款|给.*退款|drop|delete|insert|update)",
    re.I,
)


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    detail: dict | None = None

    def asdict(self):
        return {"action": self.action, "reason": self.reason, "detail": self.detail or {}}


class PolicyEngine:
    """Fail-closed checks around model-selected tools and final answers."""

    def __init__(self, allowed_merchants, max_tool_calls=16):
        self.allowed = frozenset(allowed_merchants)
        self.max_tool_calls = max_tool_calls

    def check_question(self, question):
        if WRITE_INTENT.search(question):
            return PolicyDecision("deny", "write_intent_detected")
        return PolicyDecision("allow", "question_allowed")

    def check_tool_batch(self, calls):
        if len(calls) > self.max_tool_calls:
            return PolicyDecision("deny", "too_many_tool_calls", {"count": len(calls)})
        return PolicyDecision("allow", "tool_batch_allowed", {"count": len(calls)})

    def check_tool_call(self, call):
        function = call.get("function", {})
        name = function.get("name", "")
        if name not in REGISTRY:
            return PolicyDecision("deny", "unknown_tool", {"tool": name})
        try:
            args = json.loads(function.get("arguments") or "{}")
        except ValueError:
            return PolicyDecision("deny", "invalid_tool_arguments_json", {"tool": name})
        merchant = args.get("merchant_id")
        if merchant and merchant not in self.allowed:
            return PolicyDecision("deny", "merchant_out_of_scope", {"merchant_id": merchant})
        batch = args.get("batch_id", "")
        match = re.match(r"^B\d{2}-(M\d{3})-", batch)
        if match and match.group(1) not in self.allowed:
            return PolicyDecision("deny", "batch_merchant_out_of_scope", {"batch_id": batch})
        start, end = args.get("start_date"), args.get("end_date")
        if start and end:
            try:
                days = (date.fromisoformat(end) - date.fromisoformat(start)).days
            except ValueError:
                return PolicyDecision("deny", "invalid_date_interval", {"start_date": start, "end_date": end})
            if days <= 0 or days > 366:
                return PolicyDecision("deny", "date_interval_out_of_policy", {"days": days})
        return PolicyDecision("allow", "tool_call_allowed", {"tool": name})

    def check_answer(self, answer, notebook):
        if not answer:
            return PolicyDecision("deny", "empty_answer")
        numbers = {item for item in re.findall(r"-?\d{4,}", answer)}
        known = notebook.values_for_checking()
        unsupported = sorted(n for n in numbers if n not in known and n not in {"2026", "0915", "20260915"})
        if unsupported:
            return PolicyDecision("warn", "answer_has_unverified_numbers", {"numbers": unsupported[:8]})
        return PolicyDecision("allow", "answer_supported_by_evidence")
