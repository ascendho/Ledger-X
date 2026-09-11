"""Small local-model client with streaming timing; no hosted-model fallback."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
import urllib.request

from ledger_x.specdec.model_profile import PROFILE


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://127.0.0.1:2011/v1"
    model: str = PROFILE.served_name
    timeout: int = 180

    @classmethod
    def from_env(cls):
        base_url = os.getenv("LEDGERX_BASE_URL") or cls.base_url
        model = os.getenv("LEDGERX_MODEL") or cls.model
        return cls(base_url.rstrip("/"), model)


class LocalModel:
    def __init__(self, config=None):
        self.config = config or ModelConfig.from_env()

    def chat(self, messages, tools, *, thinking=False, max_tokens=2048):
        # A trace must preserve the input at send time, not the subsequently mutated
        # conversation list (which would leak future tool results into replay data).
        payload = {"model": self.config.model, "messages": json.loads(json.dumps(messages)), "tools": tools,
                   "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                   "chat_template_kwargs": {"enable_thinking": thinking}, "stream": True,
                   "stream_options": {"include_usage": True}}
        req = urllib.request.Request(self.config.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        first = None
        calls, content, reasoning = {}, [], []
        finish, usage = None, {}
        with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                text = raw[5:].strip()
                if text == b"[DONE]":
                    break
                event = json.loads(text)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    meaningful = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls")
                    if meaningful and first is None:
                        first = time.perf_counter() - start
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    for call in delta.get("tool_calls", []):
                        item = calls.setdefault(call["index"], {"id": "", "type": "function",
                                               "function": {"name": "", "arguments": ""}})
                        if call.get("id"):
                            item["id"] = call["id"]
                        for key in ("name", "arguments"):
                            item["function"][key] += call.get("function", {}).get(key) or ""
                    finish = choice.get("finish_reason") or finish
        if finish is None:
            raise RuntimeError("Incomplete model stream")
        message = {"role": "assistant", "content": "".join(content) or None}
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return {"message": message, "finish_reason": finish, "usage": usage,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                # Semantic streamed TTFT, not a measurement of unexposed native XML tokens.
                "first_visible_delta_ms": first * 1000 if first is not None else None,
                "reasoning": "".join(reasoning), "request": payload}
