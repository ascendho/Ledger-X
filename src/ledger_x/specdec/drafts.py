"""Tokenizer-aware XML structure drafts and causal hidden-state retrieval.

This module proposes draft tokens for structured tool calls, but never masks or
modifies target logits.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import re

from ledger_x.specdec.proposer import ngram_draft


def tools_and_namespace(prompt, tokenizer_key):
    # Only the first system message is trusted as tool/context configuration.
    system = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", prompt, re.S)
    if not system:
        return {}, None
    body = system.group(1)
    block = re.search(r"<tools>\s*\n(.*?)\n</tools>", body, re.S)
    tools = {}
    if block:
        try:
            for line in block.group(1).splitlines():
                obj = json.loads(line)
                function = obj.get("function", obj)
                tools[function["name"]] = function["parameters"]
        except (ValueError, KeyError, TypeError):
            return {}, None
    marker = re.search(r"^LEDGERX_SCOPE_JSON:(\{[^\n]+\})$", body, re.M)
    if not marker:
        return tools, None  # schema drafts allowed; cross-request memory disabled
    try:
        scope = json.loads(marker.group(1))
        if not scope["tenant"] or not scope["merchants"]:
            return tools, None
        # Write policy is not part of the namespace: a frozen test can read warm memory.
        identity = {"tenant": scope["tenant"], "merchants": sorted(scope["merchants"])}
        canonical = json.dumps([tokenizer_key, tools, identity], sort_keys=True, separators=(",", ":"))
        return tools, hashlib.sha256(canonical.encode()).hexdigest()
    except (ValueError, KeyError, TypeError):
        return tools, None


def json_structural_continuation(fragment, tools):
    """Draft Qwen3's native JSON-inside-XML tool-call scaffolding."""
    start = '<tool_call>\n{"name": "'
    if start.startswith(fragment):
        return start[len(fragment):]
    if not fragment.startswith('<tool_call>\n{"name": "'):
        return ""
    rest = fragment[len('<tool_call>\n{"name": "'):]
    if '"' not in rest:
        matches = [name for name in tools if name.startswith(rest)]
        return (matches[0][len(rest):] + '", "arguments": {') if len(matches) == 1 else ""
    name, after = rest.split('"', 1)
    if name not in tools:
        return ""
    header = ', "arguments": {'
    if header.startswith(after) and after != header:
        return header[len(after):]
    if not after.startswith(header):
        return ""
    args = after[len(header):]
    schema = tools[name]
    properties = schema.get("properties", {})
    used = re.findall(r'"([^"\\]+)"\s*:', args)
    required = [key for key in schema.get("required", []) if key not in used]

    # Empty object or a completed comma is a stable property-name boundary.
    if not args or re.search(r',\s*$', args):
        key = required[0] if required else next((k for k in properties if k not in used), None)
        return f'"{key}": ' if key else '}}\n</tool_call>'

    # Complete a partially emitted property name, but never invent free-form values.
    partial_key = re.search(r'(?:^|,\s*)"([^"\\]*)$', args)
    if partial_key:
        prefix = partial_key.group(1)
        matches = [key for key in properties if key not in used and key.startswith(prefix)]
        return matches[0][len(prefix):] + '": ' if len(matches) == 1 else ""

    current = re.search(r'(?:^|,\s*)"([^"\\]+)"\s*:\s*([^,]*)$', args, re.S)
    if not current:
        return ""
    key, value = current.groups()
    spec = properties.get(key, {})
    enum = spec.get("enum", [spec["const"]] if "const" in spec else [])
    if enum:
        # Qwen JSON serializes strings with quotes and primitive values directly.
        candidates = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in enum]
        matches = [candidate for candidate in candidates if candidate.startswith(value)]
        if len(matches) == 1 and len(matches[0]) > len(value):
            return matches[0][len(value):]

    # Determine whether the current primitive value is complete. Nested arbitrary
    # objects/arrays are intentionally left to the target model.
    try:
        _, end = json.JSONDecoder().raw_decode(value)
        complete = not value[end:].strip()
    except (ValueError, TypeError):
        complete = False
    if not complete:
        return ""
    remaining = [item for item in schema.get("required", []) if item not in used]
    if remaining:
        return f', "{remaining[0]}": '
    return '}}\n</tool_call>'


def structural_continuation(text, tools):
    """Predict one possible structural continuation; uncertainty is verified later.

    Never invent a free-form value or prematurely close one. Completed tags and
    enumerated values can be extended. Already seen optional parameters are skipped.
    """
    start = text.rfind("<tool_call>")
    if start < 0:
        for tag in ('<tool_call>\n{"name": "', "<tool_call>\n<function="):
            for n in range(min(len(text), len(tag) - 1), 0, -1):
                if text.endswith(tag[:n]):
                    return tag[n:]
        return ""
    fragment = text[start:]
    if "</tool_call>" in fragment:
        return ""
    if "<function=" not in fragment:
        return json_structural_continuation(fragment, tools)
    # First complete tag may have arrived as a single vocabulary token.
    if fragment == "<tool_call>":
        return "\n<function="
    if fragment == "<tool_call>\n":
        return "<function="
    function = re.search(r"<function=([^>\n]*)", fragment)
    if not function:
        prefix = "<tool_call>\n<function="
        return prefix[len(fragment):] if prefix.startswith(fragment) else ""
    fname = function.group(1)
    function_end = function.end()
    if function_end == len(fragment):
        matches = [name for name in tools if name.startswith(fname)]
        return (matches[0][len(fname):] + ">\n") if matches else ""
    if fname not in tools:
        return ""
    schema = tools[fname]
    properties = schema.get("properties", {})
    completed = re.findall(r"<parameter=([^>]+)>\n(.*?)\n</parameter>", fragment, re.S)
    used = {name for name, _ in completed}
    # After closing a parameter, prefer an unfilled required field, then close.
    tail = fragment[function_end:]
    last_close = tail.rfind("</parameter>")
    remaining = tail[last_close + len("</parameter>"):] if last_close >= 0 else tail[1:]
    remaining = remaining.lstrip("\n")
    if remaining.startswith("</function>"):
        target = "</function>\n</tool_call>"
        return target[len(remaining):] if target.startswith(remaining) else ""
    if not remaining or "<parameter=".startswith(remaining):
        required = [name for name in schema.get("required", []) if name not in used]
        target = f"<parameter={required[0]}>\n" if required else "</function>\n</tool_call>"
        prefix = "\n" if not tail.endswith("\n") and not remaining else ""
        return prefix + target[len(remaining):] if target.startswith(remaining) else ""
    param = re.match(r"<parameter=([^>\n]*)(>?)(.*)", remaining, re.S)
    if param:
        name, closed, value = param.groups()
        if not closed:
            names = [n for n in properties if n not in used and n.startswith(name)]
            return names[0][len(name):] + ">\n" if names else ""
        spec = properties.get(name, {})
        # Partial closing tags are structural even after a free-form value.
        for tag in ("\n</parameter>\n",):
            for n in range(min(len(value), len(tag)), 1, -1):
                if value.endswith(tag[:n]):
                    return tag[n:]
        values = spec.get("enum", [spec["const"]] if "const" in spec else [])
        body = value[1:] if value.startswith("\n") else value
        for enum in values:
            target = str(enum) if isinstance(enum, str) else json.dumps(enum)
            if target.startswith(body):
                return ("\n" if not value else "") + target[len(body):] + "\n</parameter>\n"
    for tag in ("</function>\n</tool_call>",):
        if tag.startswith(remaining):
            return tag[len(remaining):]
    return ""


def encode_extension(tokenizer, accepted, continuation, budget):
    """Only draft when re-tokenization preserves the entire accepted prefix."""
    if not continuation:
        return []
    # Qwen can tokenize a delimiter together with the following identifier
    # (for example ``=query`` or ``=merchant_id``). Ending a draft at the bare
    # ``=`` creates a valid text prefix but not a token-prefix of the target
    # sequence. Leave that unstable boundary for the target model.
    if continuation.endswith("="):
        continuation = continuation[:-1]
    if not continuation:
        return []
    text = tokenizer.decode(accepted, skip_special_tokens=False)
    encoded = tokenizer.encode(text + continuation, add_special_tokens=False)
    if encoded[:len(accepted)] != list(accepted):
        return []  # BPE merged with an accepted token: cannot rewrite that token
    return encoded[len(accepted):len(accepted) + budget]


def normalized(vector):
    norm = math.sqrt(sum(float(x) * float(x) for x in vector))
    return tuple(float(x) / norm for x in vector) if norm else ()


@dataclass(frozen=True)
class MemoryEntry:
    namespace: str
    hidden: tuple
    tokens: tuple


class History:
    def __init__(self, capacity=1000):
        self.entries = deque(maxlen=capacity)

    def add(self, namespace, hidden, tokens):
        hidden = normalized(hidden)
        if namespace and hidden:
            self.entries.append(MemoryEntry(namespace, hidden, tuple(tokens)))

    def retrieve(self, namespace, hidden, top_k=10):
        query = normalized(hidden)
        if not namespace or not query:
            return []
        # numpy is already a vLLM dependency; keep the test path stdlib-only.
        eligible = [e for e in self.entries if e.namespace == namespace and len(e.hidden) == len(query)]
        try:
            import numpy as np
            scores = np.asarray([e.hidden for e in eligible], dtype=np.float32) @ np.asarray(query, dtype=np.float32) if eligible else []
        except ImportError:
            scores = [sum(a*b for a, b in zip(query, e.hidden)) for e in eligible]
        return [entry for _, entry in sorted(zip(scores, eligible), key=lambda x: -float(x[0]))[:top_k]]

    @staticmethod
    def continuation(entries, accepted, budget):
        for n in (7, 6, 5):
            if len(accepted) < n:
                continue
            suffix = tuple(accepted[-n:])
            for entry in entries:
                for start in range(len(entry.tokens) - n):
                    if entry.tokens[start:start+n] == suffix:
                        return list(entry.tokens[start+n:start+n+budget])
        return []


def _schema_valid(name, values, tools):
    if name not in tools or not isinstance(values, dict):
        return False
    try:
        import jsonschema
    except ImportError:
        return False
    try:
        jsonschema.validate(values, tools[name], format_checker=jsonschema.FormatChecker())
        return True
    except jsonschema.ValidationError:
        return False


def validate_call_text(text, tools):
    """Only complete native calls satisfying their JSON Schema enter memory."""
    raw_blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if not raw_blocks or text.count("<tool_call>") != len(raw_blocks):
        return False
    for raw in raw_blocks:
        if not raw.startswith("<function="):
            try:
                call = json.loads(raw)
            except (TypeError, ValueError):
                return False
            if set(call) != {"name", "arguments"} or not _schema_valid(
                    call["name"], call["arguments"], tools):
                return False
            continue
        match = re.fullmatch(r"<function=([^>]+)>(.*?)</function>", raw, re.S)
        if not match:
            return False
        name, body = match.groups()
        if name not in tools:
            return False
        schema, values = tools[name], {}
        matches = list(re.finditer(r"<parameter=([^>]+)>\n(.*?)\n</parameter>", body, re.S))
        residue = re.sub(r"<parameter=([^>]+)>\n(.*?)\n</parameter>", "", body, flags=re.S)
        if residue.strip():
            return False
        for match in matches:
            key, value = match.groups()
            if key in values:
                return False
            spec = schema.get("properties", {}).get(key, {})
            try:
                values[key] = value if spec.get("type") == "string" else json.loads(value)
            except ValueError:
                return False
        if not _schema_valid(name, values, tools):
            return False
    return True


class DraftEngine:
    def __init__(self, tokenizer, budget=8, mode="full"):
        self.tokenizer, self.budget, self.mode = tokenizer, budget, mode
        self.history = History()
        self.active = {}
        self.stats = Counter()
        self.tokenizer_key = hashlib.sha256((str(getattr(tokenizer, "name_or_path", "")) +
                             str(getattr(tokenizer, "chat_template", ""))).encode()).hexdigest()

    def propose(self, tokens, context):
        req_id, prompt_len = context["request_id"], context["prompt_len"]
        if req_id not in self.active:
            prompt = self.tokenizer.decode(tokens[:prompt_len], skip_special_tokens=False)
            tools, namespace = tools_and_namespace(prompt, self.tokenizer_key)
            first_system = prompt.split("<|im_end|>", 1)[0]
            marker = re.search(r"^LEDGERX_SCOPE_JSON:(\{[^\n]+\})$", first_system, re.M)
            try:
                write_memory = json.loads(marker.group(1)).get("memory_write", True) if marker else False
            except (ValueError, AttributeError):
                write_memory = False
            hidden = context.get("hidden") or []
            self.active[req_id] = {"tools": tools, "namespace": namespace, "hidden": hidden,
                "write_memory": write_memory is True,
                "retrieved": self.history.retrieve(namespace, hidden) if self.mode in {"retrieval", "full"} else [],
                "output": []}
        state = self.active[req_id]
        output = list(tokens[prompt_len:])
        state["output"] = output  # only accepted tokens, never proposed ones
        if getattr(self.tokenizer, "eos_token_id", None) in output:
            return []
        draft, source = [], "none"
        if self.mode in {"schema", "full"}:
            text = self.tokenizer.decode(output, skip_special_tokens=False)
            continuation = structural_continuation(text, state["tools"])
            draft = encode_extension(self.tokenizer, output, continuation, self.budget)
            if draft:
                source = "schema"
        if not draft and self.mode in {"retrieval", "full"}:
            draft = self.history.continuation(state["retrieved"], output, self.budget)
            if draft:
                source = "retrieval"
        if not draft and self.mode == "full":
            draft = ngram_draft(tokens, self.budget)
            if draft:
                source = "ngram"
        self.stats[source + "_tokens"] += len(draft)
        return draft

    def finish(self, request_id):
        state = self.active.pop(request_id, None)
        if not state or not state["write_memory"] or self.mode not in {"retrieval", "full"}:
            return
        output = state["output"]
        eos = getattr(self.tokenizer, "eos_token_id", None)
        # Cancellation, max-token truncation, and incomplete XML never enter memory.
        if not output or eos not in output:
            return
        # vLLM may accept a block containing tokens after EOS; the scheduler exposes
        # only the prefix ending at EOS. Match that boundary before storing history.
        output = output[:output.index(eos) + 1]
        text = self.tokenizer.decode(output[:-1], skip_special_tokens=False)
        if validate_call_text(text, state["tools"]):
            self.history.add(state["namespace"], state["hidden"], output)
