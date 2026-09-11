"""Small investigation playbooks used as planning hints, not hard-coded answers."""
from __future__ import annotations


PLAYBOOKS = {
    "exception_triage": {
        "required_tools": ["list_reconciliation_exceptions", "reconcile_settlement_batch"],
        "hint": "先列出异常批次，再复算目标批次，最后按工具证据解释原因。",
    },
    "batch_audit": {
        "required_tools": ["reconcile_settlement_batch"],
        "hint": "直接复算批次，报告应结算、实际到账、差额和异常证据。",
    },
    "transaction_trace": {
        "required_tools": ["get_transaction_detail"],
        "hint": "追踪交易、退款和所属结算批次，不推断工具外事实。",
    },
    "fee_rule": {
        "required_tools": ["get_settlement_rule"],
        "hint": "按交易日期查询费率规则，回答规则编号和费率。",
    },
    "summary": {
        "required_tools": ["query_settlement_summary"],
        "hint": "按要求汇总结算金额、到账金额和差额。",
    },
}


def select_playbook(question):
    q = question.lower()
    if "交易" in question or "transaction" in q or "T0" in question:
        name = "transaction_trace"
    elif "费率" in question or "手续费规则" in question:
        name = "fee_rule"
    elif "批次" in question and ("复算" in question or "核算" in question or "重新" in question):
        name = "batch_audit"
    elif "汇总" in question or "按渠道" in question or "按天" in question:
        name = "summary"
    elif "异常" in question or "差额" in question:
        name = "exception_triage"
    else:
        name = "exception_triage"
    return {"name": name, **PLAYBOOKS[name]}
