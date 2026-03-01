#!/usr/bin/env python3
# utils/concept_extractor.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("concepts")

# --- knobs (env-overridable) -------------------------------------------------

MAX_SURFACES_PER_STEP = int(os.getenv("CE_MAX_SURFACES_PER_STEP", "12"))
MAX_CANDIDATES_PER_STEP = int(os.getenv("CE_MAX_CANDIDATES_PER_STEP", "32"))
NGRAM_MIN = 2
NGRAM_MAX = 5

# Light stopword list (mirrors umls_api_linker enough for surface generation)
STOPWORDS = {
    "a","an","the","of","on","in","to","is","are","and","with","for","as","at","by","or","from",
    "via","into","than","that","this","those","these","be","been","being","was","were","will","would",
    "can","could","should","may","might","not","no","yes","it","its","their","there","then","thus",
    "we","our","you","your","i","he","she","they","them","his","her"
}

# --- helpers -----------------------------------------------------------------

_SPLIT_PUNCT = re.compile(r"[.;:!?]|(?:\s-\s)")
_PARENS = re.compile(r"\(([^)]+)\)")
_ACRO = re.compile(r"\b[A-Z]{2,6}\b")
_TOKEN_SPLIT = re.compile(r"[\s/,-]+")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _is_trivial(surface: str) -> bool:
    t = surface.lower()
    if not t or len(t) <= 1: return True
    if t in STOPWORDS: return True
    if t.startswith(("final answer", "final:", "answer:")): return True
    if re.fullmatch(r"\W+", t): return True
    return False

def _tokens(s: str) -> List[str]:
    parts = [re.sub(r"^[^\w]+|[^\w]+$", "", p) for p in _TOKEN_SPLIT.split(s)]
    return [p for p in parts if p and p.lower() not in STOPWORDS]

def _ngrams(ts: List[str], nmin=NGRAM_MIN, nmax=NGRAM_MAX) -> List[str]:
    out: List[str] = []
    L = len(ts)
    if L < nmin: return out
    for n in range(nmin, min(nmax, L) + 1):
        for i in range(0, L - n + 1):
            phrase = " ".join(ts[i:i+n])
            if not _is_trivial(phrase):
                out.append(phrase)
    # dedupe preserving order
    seen, uniq = set(), []
    for p in out:
        k = p.lower()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq

def _surface_candidates_from_step(step: str) -> List[str]:
    s = _norm(step)
    if not s:
        return []
    cands: List[str] = []

    # 1) contents in parentheses (often abbreviations or specific entities)
    for inside in _PARENS.findall(s):
        inside = _norm(inside)
        if inside and not _is_trivial(inside):
            cands.append(inside)

    # 2) split by major punctuation to get shorter phrases
    for chunk in _SPLIT_PUNCT.split(s):
        chunk = _norm(chunk)
        if chunk and not _is_trivial(chunk):
            cands.append(chunk)

    # 3) acronyms (DBE, T2DM, MI, etc.)
    for ac in _ACRO.findall(s):
        if not _is_trivial(ac):
            cands.append(ac)

    # 4) hyphenated/compound words are kept by the tokenizer; add n-grams
    ts = _tokens(s)
    cands.extend(_ngrams(ts, NGRAM_MIN, NGRAM_MAX))

    # 5) keep some full phrases if they look biomedical-ish (contain a long token)
    if any(len(t) >= 6 for t in ts):
        cands.append(s)

    # dedupe, cap
    seen, uniq = set(), []
    for v in cands:
        vv = _norm(v)
        k = vv.lower()
        if vv and not _is_trivial(vv) and k not in seen:
            seen.add(k); uniq.append(vv)
        if len(uniq) >= MAX_SURFACES_PER_STEP:
            break
    return uniq

# --- UMLS linker integration --------------------------------------------------

def _umls_is_configured() -> bool:
    try:
        from utils.umls_api_linker import is_configured  # type: ignore
        return bool(is_configured())
    except Exception:
        return False

def _link_batch(
    surfaces: List[str],
    *,
    top_k: int,
    scispacy_when: str,
    allowed_kb_sources: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Call umls_api_linker.link_texts_batch with graceful fallback for unknown kwargs.
    Always returns a dict {surface: [candidates...]}.
    """
    from utils import umls_api_linker as linker  # late import to avoid cycles
    # Try with 'parallel' kw if the installed linker supports it;
    # fall back to a plain call when not.
    try:
        # Some older/newer builds expose a 'parallel' flag — not required.
        return linker.link_texts_batch(
            surfaces,
            top_k=top_k,
            scispacy_when=scispacy_when,
            allowed_kb_sources=allowed_kb_sources,
            parallel=True,  # will raise TypeError on versions that don't accept it
        )
    except TypeError:
        # Retry without the unknown kwarg (silences: "got an unexpected keyword argument 'parallel'")
        return linker.link_texts_batch(
            surfaces,
            top_k=top_k,
            scispacy_when=scispacy_when,
            allowed_kb_sources=allowed_kb_sources,
        )

# --- public API ---------------------------------------------------------------

def extract_concepts(
    steps: List[str],
    *,
    scispacy_when: str = "auto",
    top_k: int = 3,
    top_k_umls: Optional[int] = None,
    allowed_kb_sources: Optional[List[str]] = None,
    **kwargs: Any,   # accepts stray/legacy kwargs (e.g., 'parallel') without exploding
) -> List[List[Dict[str, Any]]]:
    """
    Returns: list-of-lists aligned to steps.
      per_step[i] is a flat list of candidate dicts (merged across surfaces for that step).

    Each candidate dict (as produced by umls_api_linker) looks like:
      {
        "text": <original surface>,
        "cui": "Cxxxxxx" | None,
        "canonical": "...",
        "semantic_types": [...],
        "kb_sources": [...],
        "valid": bool,
        "scores": {"api": float | None, "link": float | None, "confidence": float}
      }
    """
    K = int(top_k_umls or top_k or 3)

    if not steps:
        return []

    if not _umls_is_configured():
        log.info("[concepts] UMLS not configured; returning empty concept sets.")
        return [[] for _ in steps]

    # 1) generate surfaces per step
    per_step_surfaces: List[List[str]] = [ _surface_candidates_from_step(s or "") for s in steps ]
    # Build a global de-duplicated list to query once
    seen, all_surfaces = set(), []
    for arr in per_step_surfaces:
        for s in arr:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                all_surfaces.append(s)

    # Edge: nothing extracted → return empties
    if not all_surfaces:
        return [[] for _ in steps]

    # 2) link (batch)
    try:
        by_surface = _link_batch(
            all_surfaces,
            top_k=K,
            scispacy_when=scispacy_when,
            allowed_kb_sources=allowed_kb_sources,
        )
    except Exception as e:
        log.warning("[concepts] link_texts_batch failed: %s", e)
        return [[] for _ in steps]

    # 3) map back to steps (merge candidates from all surfaces of the step)
    per_step_out: List[List[Dict[str, Any]]] = []
    linked_count = 0
    for surfaces in per_step_surfaces:
        acc: List[Dict[str, Any]] = []
        for s in surfaces:
            lst = list(by_surface.get(s, []) or [])
            if lst:
                linked_count += 1
            acc.extend(lst)
        # sort by blended confidence, cap
        acc.sort(key=lambda r: float(((r.get("scores") or {}).get("confidence")) or 0.0), reverse=True)
        if len(acc) > MAX_CANDIDATES_PER_STEP:
            acc = acc[:MAX_CANDIDATES_PER_STEP]
        per_step_out.append(acc)

    log.info("[concepts] API-only mode: surfaces=%d, linked_surfaces=%d", len(all_surfaces), linked_count)

    # 4) Validate concepts via UMLS checker (semantic types, relations, score thresholds)
    try:
        from utils.umls_checker import make_checker, validate_concepts as _val_concepts
        checker = make_checker(enable_relation_check=False)
        per_step_out = _val_concepts(per_step_out, checker=checker)
        valid_count = sum(1 for step in per_step_out for c in step if c.get("valid"))
        total_count = sum(len(step) for step in per_step_out)
        log.info("[concepts] UMLS validation: %d/%d concepts valid", valid_count, total_count)
    except Exception as e:
        log.debug("[concepts] UMLS validation skipped: %s", e)

    return per_step_out
