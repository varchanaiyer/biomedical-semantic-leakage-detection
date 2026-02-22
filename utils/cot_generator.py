# utils/cot_generator.py
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Config & readiness
# -----------------------------------------------------------------------------

def _cfg(name: str):
    """Import a single name from config, falling back to env var silently."""
    try:
        import config as _cfg_mod
        return getattr(_cfg_mod, name, None) or None
    except Exception:
        return None

ANTHROPIC_API_KEY  = _cfg("ANTHROPIC_API_KEY")  or os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY     = _cfg("OPENAI_API_KEY")      or os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY     = _cfg("GOOGLE_API_KEY")      or os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = _cfg("OPENROUTER_API_KEY")  or os.getenv("OPENROUTER_API_KEY")

ANTHROPIC_READY   = bool(ANTHROPIC_API_KEY)
OPENAI_READY      = bool(OPENAI_API_KEY)
GEMINI_READY      = bool(GOOGLE_API_KEY)
OPENROUTER_READY  = bool(OPENROUTER_API_KEY)

ANTHROPIC_MODEL_DEFAULT    = "claude-haiku-4-5"
OPENAI_MODEL_DEFAULT       = "gpt-4o-mini"
OPENAI_O4_MODEL            = "o4-mini"
GEMINI_MODEL_DEFAULT       = "gemini-1.5-flash"
# OpenRouter: use claude-haiku via OR by default; override with OPENROUTER_MODEL env var
OPENROUTER_MODEL_DEFAULT   = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4-5")
OPENROUTER_BASE_URL        = "https://openrouter.ai/api/v1"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_LEADING_MARK_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-•]\s*)")

def _postprocess_steps(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    steps: List[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        ln = _LEADING_MARK_RE.sub("", ln).strip()
        if ln:
            steps.append(ln)
    if len(steps) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", raw) if p.strip()]
        if len(parts) > 1:
            steps = parts
    return steps or ["Identify key entities", "Map relations", "Synthesize answer"]

def _json_safe(obj: Any) -> Any:
    """Ensure object can be JSON-serialized."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def _mk_result(
    *,
    steps: List[str],
    provider: str,
    model: str,
    final: str = "",
    meta: Optional[Dict[str, Any]] = None,
    raw: Optional[Any] = None,
) -> Dict[str, Any]:
    return {
        "steps": steps,
        "final": final or "",
        "provider": provider,
        "model": model,
        "meta": _json_safe(meta or {}),
        "raw": _json_safe(raw),
    }

# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------

def _call_anthropic(question: str) -> Optional[Dict[str, Any]]:
    if not ANTHROPIC_API_KEY:
        return None
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL_DEFAULT,
            max_tokens=700,
            temperature=0.2,
            system="Return concise numbered reasoning steps only.",
            messages=[{
                "role": "user",
                "content": f"Break down the reasoning into short numbered steps:\n{question}"
            }],
        )
        text = "".join(getattr(p, "text", "") for p in (getattr(msg, "content", []) or []))
        steps = _postprocess_steps(text)
        return _mk_result(
            steps=steps,
            provider="anthropic",
            model=ANTHROPIC_MODEL_DEFAULT,
            meta={"token_usage": _json_safe(getattr(msg, "usage", None))},
            raw=msg,
        )
    except Exception:
        return None

def _call_openai(question: str, *, prefer_o4: bool) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        model = OPENAI_O4_MODEL if prefer_o4 else OPENAI_MODEL_DEFAULT
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=600,
            messages=[
                {"role": "system", "content": "Return concise numbered reasoning steps only."},
                {"role": "user", "content": f"Break down the reasoning into short numbered steps:\n{question}"},
            ],
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        steps = _postprocess_steps(text)
        return _mk_result(
            steps=steps,
            provider="openai",
            model=model,
            meta={"usage": _json_safe(getattr(resp, "usage", None))},
            raw=resp,
        )
    except Exception:
        return None

def _call_openrouter(question: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Call any model via OpenRouter's OpenAI-compatible endpoint."""
    if not OPENROUTER_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        _model = model or OPENROUTER_MODEL_DEFAULT
        resp = client.chat.completions.create(
            model=_model,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {"role": "system", "content": "Return concise numbered reasoning steps only."},
                {"role": "user", "content": f"Break down the reasoning into short numbered steps:\n{question}"},
            ],
            extra_headers={
                "HTTP-Referer": "https://github.com/biomedical-semantic-leakage",
                "X-Title": "Biomedical Semantic Leakage Detection",
            },
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        steps = _postprocess_steps(text)
        return _mk_result(
            steps=steps,
            provider="openrouter",
            model=_model,
            meta={"usage": _json_safe(getattr(resp, "usage", None))},
            raw=resp,
        )
    except Exception:
        return None


def _call_gemini(question: str) -> Optional[Dict[str, Any]]:
    if not GOOGLE_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL_DEFAULT)
        prompt = f"Return concise numbered reasoning steps only.\n\nQuestion:\n{question}"
        resp = model.generate_content(prompt)
        text = ""
        try:
            text = resp.text
        except Exception:
            try:
                text = " ".join([p.text for p in getattr(resp, "candidates", [])])  # type: ignore
            except Exception:
                text = ""
        steps = _postprocess_steps(text)
        return _mk_result(
            steps=steps,
            provider="gemini",
            model=GEMINI_MODEL_DEFAULT,
            meta={"prompt_feedback": _json_safe(getattr(resp, "prompt_feedback", None))},
            raw=resp,
        )
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate(question: str, prefer: str = "openrouter", model: Optional[str] = None) -> Dict[str, Any]:
    """Generate CoT steps for a question.

    Args:
        question: The biomedical question to reason about.
        prefer:   Provider preference ('openrouter', 'anthropic', 'openai', 'gemini').
        model:    Optional specific model ID to pass to the provider.
                  When prefer='openrouter' this is an OpenRouter model slug, e.g.
                  'anthropic/claude-haiku-4-5', 'openai/gpt-4o-mini',
                  'google/gemini-flash-1.5', 'meta-llama/llama-3.3-70b-instruct'.
    """
    prefer = (prefer or "").lower().strip()

    # If a specific model is requested, route directly through OpenRouter
    if model:
        res = _call_openrouter(question, model=model)
        if res and res.get("steps"):
            return res
        # fall through to normal order below

    order = {
        "openrouter": (_call_openrouter, _call_anthropic, lambda q: _call_openai(q, prefer_o4=False), _call_gemini),
        "anthropic":  (_call_anthropic, _call_openrouter, lambda q: _call_openai(q, prefer_o4=True), lambda q: _call_openai(q, prefer_o4=False), _call_gemini),
        "openai":     (lambda q: _call_openai(q, prefer_o4=True), lambda q: _call_openai(q, prefer_o4=False), _call_openrouter, _call_anthropic, _call_gemini),
        "o4":         (lambda q: _call_openai(q, prefer_o4=True), lambda q: _call_openai(q, prefer_o4=False), _call_openrouter, _call_anthropic, _call_gemini),
        "gemini":     (_call_gemini, _call_openrouter, _call_anthropic, lambda q: _call_openai(q, prefer_o4=True), lambda q: _call_openai(q, prefer_o4=False)),
    }.get(prefer, (_call_openrouter, _call_anthropic, lambda q: _call_openai(q, prefer_o4=False), _call_gemini))

    for fn in order:
        res = fn(question)
        if res and res.get("steps"):
            return res

    return _mk_result(
        steps=[
            "Identify key biomedical entities and mechanisms in the question.",
            "Recall guideline-backed treatments and mechanisms of action.",
            "Map drugs to targets and diseases to relevant pathophysiology.",
            "Consider contraindications and important subgroups.",
            "Synthesize a mechanism-to-outcome reasoning path.",
        ],
        provider="local",
        model="fallback-rules",
        final="",
        meta={"note": "All provider calls failed; using heuristic steps."},
        raw=None,
    )

def generate_cot(question: str, prefer: str = "anthropic") -> Dict[str, Any]:
    return generate(question, prefer=prefer)

def run(question: str, prefer: str = "anthropic") -> Dict[str, Any]:
    return generate(question, prefer=prefer)

def demo(question: str, prefer: str = "anthropic") -> Dict[str, Any]:
    return generate(question, prefer=prefer)

def produce(question: str, prefer: str = "anthropic") -> Dict[str, Any]:
    return generate(question, prefer=prefer)

class CoTGenerator:
    def __init__(self, prefer: str = "anthropic"):
        self.prefer = prefer

    def generate(self, question: str) -> Dict[str, Any]:
        return generate(question, prefer=self.prefer)
