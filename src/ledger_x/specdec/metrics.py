"""Derived metrics for Retrieval-Augmented Speculation activity."""
from __future__ import annotations


def counter_delta(before, after):
    """Subtract cumulative numeric counters and clamp process-reset artifacts."""
    keys = set(before) | set(after)
    return {key: max(0, after.get(key, 0) - before.get(key, 0))
            for key in keys
            if isinstance(before.get(key, 0), (int, float))
            and isinstance(after.get(key, 0), (int, float))}


def retrieval_summary(counters):
    """Derive RAS activity metrics without mislabeling availability as recall."""
    eligible = counters.get("retrieval_eligible_requests", 0)
    candidates = counters.get("retrieval_candidate_requests", 0)
    drafted = counters.get("retrieval_draft_requests", 0)
    proposed = counters.get("proposed_tokens", 0)
    retrieval_tokens = counters.get("retrieval_tokens", 0)
    return {
        "eligible_requests": eligible,
        "candidate_requests": candidates,
        "candidate_availability_at_10": candidates / eligible if eligible else None,
        "draft_requests": drafted,
        "draft_request_coverage": drafted / eligible if eligible else None,
        "draft_steps": counters.get("retrieval_draft_steps", 0),
        "proposed_tokens": retrieval_tokens,
        "retrieval_token_share": retrieval_tokens / proposed if proposed else None,
        "note": ("Candidate availability and draft coverage are runtime activity metrics, "
                 "not information-retrieval Recall@K."),
    }
