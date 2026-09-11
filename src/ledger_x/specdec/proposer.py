"""Single-chain, target-verified drafts. No financial result caching.

The empty/wrong modes are deliberately diagnostic, never deployment defaults.
The hidden-state bridge is required for retrieval, not inferred from HTTP JSON.
"""
from __future__ import annotations

import os
import time
from collections import Counter
import json
from pathlib import Path


def ngram_draft(tokens, budget, min_n=5, max_n=7):
    tokens = list(tokens)
    for n in range(min(max_n, len(tokens)), min_n - 1, -1):
        suffix = tokens[-n:]
        for start in range(len(tokens) - n - 1, -1, -1):
            if tokens[start:start + n] == suffix:
                return tokens[start + n:min(start + n + budget, len(tokens))]
    return []


class LedgerXProposer:
    def __init__(self, vllm_config):
        spec = vllm_config.speculative_config
        self.budget = spec.num_speculative_tokens
        self.max_len = vllm_config.model_config.max_model_len
        self.mode = os.environ.get("LEDGERX_DRAFT_MODE", "empty")
        if self.mode not in {"empty", "wrong", "ngram", "schema", "retrieval", "full"}:
            raise ValueError("Unknown LEDGERX_DRAFT_MODE")
        self.stats = Counter()
        self.stats_path = os.environ.get("LEDGERX_STATS_PATH")
        self.engine = None
        if self.mode in {"schema", "retrieval", "full"}:
            from transformers import AutoTokenizer
            from ledger_x.specdec.drafts import DraftEngine
            tokenizer = AutoTokenizer.from_pretrained(vllm_config.model_config.model,
                                                       local_files_only=True)
            self.engine = DraftEngine(tokenizer, self.budget, self.mode)

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
                slot_mappings=None, ledger_x_context=None):
        start = time.perf_counter()
        result = []
        if self.engine and ledger_x_context is None:
            raise RuntimeError("Schema/retrieval mode requires the version-pinned runner bridge")
        for row, sampled in enumerate(sampled_token_ids):
            length = int(num_tokens_no_spec[row])
            budget = min(self.budget, self.max_len - length)
            if not sampled or budget <= 0:
                result.append([])
                continue
            # vLLM's synchronous runner has already appended accepted samples.
            tokens = token_ids_cpu[row, :length].tolist()
            if self.mode == "empty":
                draft = []
            elif self.mode == "wrong":
                draft = [0] * budget
            elif self.mode == "ngram":
                draft = ngram_draft(tokens, budget)
            else:
                draft = self.engine.propose(tokens, ledger_x_context[row])[:budget]
            result.append(draft)
            self.stats["proposed_tokens"] += len(draft)
            self.stats["rows"] += 1
            self.stats[self.mode + "_rows"] += 1
            if self.engine:
                self.stats.update(self.engine.stats)
                self.engine.stats.clear()
            elif draft:
                self.stats[self.mode + "_tokens"] += len(draft)
            else:
                self.stats["empty_rows"] += 1
        self.stats["calls"] += 1
        self.stats["propose_seconds"] += time.perf_counter() - start
        self._export_stats()
        return result

    def _export_stats(self):
        if not self.stats_path:
            return
        path = Path(self.stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": self.mode, "budget": self.budget, "counters": dict(self.stats),
                   "note": "Counts draft tokens returned by the proposer. Accepted-token counts must come from vLLM verification metrics."}
        # TP workers are separate processes and may export simultaneously. A
        # process-specific temporary name avoids one worker replacing another's
        # temporary file before it can complete its own atomic replace.
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
