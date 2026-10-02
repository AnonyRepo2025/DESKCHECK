"""Records the pipelines persist, built from the fields they need.

Provider payloads (litellm responses, Claude Code stream-json events) are never copied into a
record wholesale. Each record below is constructed field by field from an explicit list of what
the pipeline and its accounting read, so anything a provider adds -- including prompt-caching
detail -- never reaches disk.

Usage records hold exactly what total-token and USD-cost accounting needs:

* OpenAI / litellm shape (mini-swe-agent pipeline):
  ``{"prompt_tokens", "completion_tokens", "total_tokens", "cost"}``
* Anthropic / Claude Code shape (ccpipe):
  ``{"input_tokens", "output_tokens", "total_tokens"[, "cost"]}``

Input/prompt counts are ALL input tokens: cached and freshly-processed input are counted alike.
Every builder is idempotent -- building from an already-built record returns an equal record.
"""
from __future__ import annotations


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _get(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _pick(src: dict, keys) -> dict:
    """The listed keys of ``src`` that are present (value copied as-is)."""
    return {k: src[k] for k in keys if isinstance(src, dict) and k in src}


# -------------------------------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------------------------------
def openai_usage(u, cost=None) -> dict:
    """litellm/OpenAI ``usage`` (dict or object) -> prompt/completion/total tokens + cost.

    ``prompt_tokens`` already includes cached input in this convention. ``cost`` falls back to
    the usage block's own ``cost`` (OpenRouter reports it there)."""
    u = {} if u is None else u
    pt = _int(_get(u, "prompt_tokens")) or _int(_get(u, "input_tokens"))        # Responses-API naming
    ct = _int(_get(u, "completion_tokens")) or _int(_get(u, "output_tokens"))
    c = _float(cost) if cost is not None else _float(_get(u, "cost"))
    return {"prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": _int(_get(u, "total_tokens")) or pt + ct, "cost": c or 0.0}


def anthropic_usage(u, cost=None) -> dict:
    """Anthropic ``usage`` (snake_case) or a Claude Code ``modelUsage`` entry (camelCase) ->
    input/output/total tokens (+ cost when known). Input = every input token the call processed
    (fresh + written to cache + read from cache)."""
    u = u or {}
    inp = (_int(u.get("input_tokens")) + _int(u.get("cache_creation_input_tokens"))
           + _int(u.get("cache_read_input_tokens"))
           + _int(u.get("inputTokens")) + _int(u.get("cacheCreationInputTokens"))
           + _int(u.get("cacheReadInputTokens")))
    out = _int(u.get("output_tokens")) + _int(u.get("outputTokens"))
    rec = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    c = _float(cost) if cost is not None else _float(u.get("cost", u.get("costUSD")))
    if c is not None:
        rec["cost"] = c
    return rec


def model_usage_total(model_usage: dict) -> dict:
    """Sum a Claude Code ``modelUsage`` map ({model: counters}) into one anthropic_usage record.
    This is the step's complete aggregate (the result event's own ``usage`` can cover only part
    of the step)."""
    tot = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}
    for m in (model_usage or {}).values():
        r = anthropic_usage(m)
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            tot[k] += r[k]
        tot["cost"] += r.get("cost") or 0.0
    return tot


# -------------------------------------------------------------------------------------------
# mini-swe-agent: the provider response kept on a message (extra["response"])
# -------------------------------------------------------------------------------------------
# Read from a persisted response: the assistant text (FormatError recovery,
# reproduce_intervention_subagent._text_from_format_error) and the usage (token accounting).
_RESPONSE_MESSAGE_KEYS = ("role", "content", "reasoning_content", "reasoning")


def openai_response_record(resp, cost=None) -> dict:
    """``{"choices": [{"message": {role, content, reasoning_content, reasoning}}], "usage": ...}``
    built from a litellm response (``model_dump`` dict or object)."""
    choices = []
    for ch in (_get(resp, "choices") or []) if resp is not None else []:
        msg = _get(ch, "message")
        if msg is not None and not isinstance(msg, dict):
            msg = {k: getattr(msg, k) for k in _RESPONSE_MESSAGE_KEYS if hasattr(msg, k)}
        choices.append({"message": _pick(msg or {}, _RESPONSE_MESSAGE_KEYS)})
    return {"choices": choices,
            "usage": openai_usage(_get(resp, "usage") if resp is not None else None, cost=cost)}


# -------------------------------------------------------------------------------------------
# ccpipe: Claude Code stream-json events
# -------------------------------------------------------------------------------------------
# Exactly the fields ccpipe reads back from a stream (traj.convert, validate_cc.literal_rewrites).
_CC_EVENT_KEYS = ("type", "subtype", "session_id", "uuid", "parent_tool_use_id", "model")
_CC_MESSAGE_KEYS = ("id", "role", "model", "content", "stop_reason")
_CC_RESULT_KEYS = ("num_turns", "total_cost_usd", "result", "is_error", "structured_output",
                   "stop_reason", "terminal_reason", "permission_denials")


def cc_event_record(ev: dict) -> dict:
    """One persisted stream event, built from the fields ccpipe reads.

    Message ``content`` (text, thinking, tool calls and tool results) is the transcript and is
    kept verbatim. Usage is built from token counts: per assistant message tokens only; for the
    result event, the step's complete aggregate (``modelUsage`` total) with ``total_cost_usd``."""
    if not isinstance(ev, dict):
        return ev
    rec = _pick(ev, _CC_EVENT_KEYS)
    msg = ev.get("message")
    if isinstance(msg, dict):
        rec["message"] = _pick(msg, _CC_MESSAGE_KEYS)
        if isinstance(msg.get("usage"), dict):
            rec["message"]["usage"] = anthropic_usage(msg["usage"])
    if ev.get("type") == "result":
        rec.update(_pick(ev, _CC_RESULT_KEYS))
        if isinstance(ev.get("modelUsage"), dict) and ev["modelUsage"]:
            usage = model_usage_total(ev["modelUsage"])
            if ev.get("total_cost_usd") is not None:
                usage["cost"] = float(ev["total_cost_usd"])
        else:
            usage = anthropic_usage(ev.get("usage"), cost=ev.get("total_cost_usd"))
        rec["usage"] = usage
    return rec
