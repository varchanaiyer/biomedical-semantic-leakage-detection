#!/usr/bin/env python3
# utils/umls_api_linker.py
# -*- coding: utf-8 -*-
"""
UMLS Toolkit — apiKey mode (no CAS), robust + high-coverage + connection-safe.

Key improvements for your workload:
• Accepts parallel=True in link_texts_batch(...) (internal bounded concurrency).
• Larger requests pool (configurable) to stop "connection pool is full" warnings.
• Global in-flight cap + jittered backoff to be kind to UMLS (and avoid timeouts).
• Variant + n-gram expansion (coverage), but with noise filtering (e.g., "around 60-70%").
• Multi-search strategy: default → words → approximate; results merged by CUI.
• Disk+memory cache and LRU caching of /search results.

ENV knobs (override defaults as needed):
  UMLS_API_KEY                 your key (do NOT hardcode)
  UMLS_REST_ENDPOINT           defaults to https://uts-ws.nlm.nih.gov
  UMLS_REQ_TIMEOUT             per-request read timeout (sec, default 7)
  UMLS_CONNECT_TIMEOUT         connect timeout (sec, default 3)
  UMLS_MAX_RETRIES             retry count for transient errors (default 3)
  UMLS_HTTP_POOL               requests pool_maxsize per adapter (default 64)
  UMLS_MAX_INFLIGHT            global concurrent HTTP calls (default 24)
  UMLS_DEFAULT_SABS            e.g. "MSH,SNOMEDCT_US,RXNORM"
  UMLS_DEFAULT_TTYS            e.g. "PT,SY,IN"
  UMLS_CACHE_DISABLE           "1"/"true" to disable disk cache
  UMLS_NOISE_MAX_TOKENS        max tokens per query before skip (default 8)
  UMLS_NOISE_MIN_ALPHA         require ≥ this many alphabetic chars (default 3)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
import random
from functools import lru_cache
from threading import RLock, BoundedSemaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# ============================== Config =======================================

try:
    from config import (
        UMLS_API_KEY,
        UMLS_USERNAME,        # unused in apiKey mode; kept for compatibility
        UMLS_REST_ENDPOINT,   # optional override
        UMLS_AUTH_ENDPOINT,   # unused in apiKey mode; kept for compatibility
    )
except Exception:
    UMLS_API_KEY = os.getenv("UMLS_API_KEY", "")
    UMLS_USERNAME = os.getenv("UMLS_USERNAME", "")
    UMLS_REST_ENDPOINT = os.getenv("UMLS_REST_ENDPOINT", "https://uts-ws.nlm.nih.gov")
    UMLS_AUTH_ENDPOINT = os.getenv("UMLS_AUTH_ENDPOINT", "https://utslogin.nlm.nih.gov")

def is_configured() -> bool:
    key_ok = bool(UMLS_API_KEY) and UMLS_API_KEY.strip().lower() not in {"", "your_key_here", "changeme"}
    rest_ok = bool(UMLS_REST_ENDPOINT)
    return key_ok and rest_ok

UTS_REST = f"{UMLS_REST_ENDPOINT.rstrip('/')}/rest"
DEFAULT_VERSION = "current"

# Defaults + ENV
REQ_TIMEOUT = float(os.getenv("UMLS_REQ_TIMEOUT", "7"))
CONNECT_TIMEOUT = float(os.getenv("UMLS_CONNECT_TIMEOUT", "3"))
MAX_RETRIES = int(os.getenv("UMLS_MAX_RETRIES", "3"))
HTTP_POOL = int(os.getenv("UMLS_HTTP_POOL", "64"))
MAX_INFLIGHT = int(os.getenv("UMLS_MAX_INFLIGHT", "24"))
DEFAULT_SABS = os.getenv("UMLS_DEFAULT_SABS", "")  # e.g. "MSH,SNOMEDCT_US,RXNORM"
DEFAULT_TTYS = os.getenv("UMLS_DEFAULT_TTYS", "")  # e.g. "PT,SY,IN"
NOISE_MAX_TOKENS = int(os.getenv("UMLS_NOISE_MAX_TOKENS", "8"))
NOISE_MIN_ALPHA = int(os.getenv("UMLS_NOISE_MIN_ALPHA", "3"))

# Coverage tuning (reduced for speed; original: 8/12)
MAX_VARIANTS_PER_SURFACE = 4
MAX_NGRAMS_PER_SURFACE   = 6
MIN_NGRAM_LEN            = 2
MAX_NGRAM_LEN            = 4

# ============================== Logging ======================================

log = logging.getLogger("umls")

# ============================== HTTP session =================================

SES = requests.Session()
SES.headers.update({
    "User-Agent": "UMLS-Toolkit/apikey",
    "Connection": "keep-alive",
    "Accept": "application/json",
})

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry  # type: ignore

    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.4,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=HTTP_POOL, pool_maxsize=HTTP_POOL)
    SES.mount("https://", adapter)
    SES.mount("http://", adapter)
except Exception:
    pass

# Global inflight limiter to avoid pool floods / rate spikes
_INFLIGHT = BoundedSemaphore(MAX_INFLIGHT)

# ============================== Cache (disk + mem) ===========================

CACHE_PATH = os.path.join(os.path.dirname(__file__), ".umls_cache.json")
CACHE_DISABLE = os.getenv("UMLS_CACHE_DISABLE", "").lower() in {"1", "true", "yes"}
CACHE_SAVE_INTERVAL = 100  # save every N inserts

_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = RLock()

if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)
    except Exception:
        _CACHE = {}

def _save_cache() -> None:
    if CACHE_DISABLE:
        return
    try:
        with _CACHE_LOCK:
            snap = copy.deepcopy(_CACHE)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        os.replace(tmp, CACHE_PATH)  # atomic on POSIX
    except Exception as e:
        log.warning("Failed to save UMLS cache: %s", e)

# ============================== Helpers ======================================

STOPWORDS = {
    "a","an","the","of","on","in","to","is","are","and","with","for","as","at","by","or","from",
    "via","into","than","that","this","those","these","be","been","being","was","were","will","would",
    "can","could","should","may","might","not","no","yes","it","its","their","there","then","thus",
    "we","our","you","your","i","he","she","they","them","his","her"
}

def _backoff(attempt: int) -> None:
    # exponential with jitter
    base = min(1.6 ** attempt, 6.0)
    time.sleep(base * (0.5 + random.random() * 0.7))

def _ensure_key(args_key: Optional[str]) -> str:
    apikey = (args_key or "").strip() or UMLS_API_KEY or os.getenv("UMLS_API_KEY", "")
    if not apikey:
        raise SystemExit("Missing UMLS_API_KEY (env or config.py).")
    return apikey

def _inject_key(params: Dict[str, Any], apikey: str) -> Dict[str, Any]:
    q = dict(params or {})
    # never log/echo the key
    q["apiKey"] = apikey
    return q

def _api_get(path: str, params: Dict[str, Any]) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with _INFLIGHT:  # cap concurrent calls globally
                r = SES.get(
                    f"{UTS_REST}{path}",
                    params=params,
                    timeout=(CONNECT_TIMEOUT, REQ_TIMEOUT),
                )
            # Retry on soft failures guided by Retry adapter; here handle manual cases too.
            if r.status_code in (408, 429, 500, 502, 503, 504):
                _backoff(attempt); continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            _backoff(attempt)
    raise RuntimeError(f"GET {path} failed after retries: {last_exc}")

def _is_trivial(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t: return True
    if len(t) <= 1: return True
    if t in STOPWORDS: return True
    if re.fullmatch(r"\W+", t): return True
    return False

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _tokenize_for_ngrams(s: str) -> List[str]:
    toks = re.split(r"[\s\-/]+", s.strip())
    clean = []
    for t in toks:
        tt = re.sub(r"^[^\w]+|[^\w]+$", "", t)
        if not tt: continue
        if tt.lower() in STOPWORDS: continue
        clean.append(tt)
    return clean

# Noise filter: skip boilerplate-y fragments that waste quota / timeout
_PCT = re.compile(r"%")
def _looks_like_noise(term: str) -> bool:
    s = _normalize_ws(term)
    if not s: return True
    if _PCT.search(s):  # "around 60-70%"
        return True
    toks = _tokenize_for_ngrams(s)
    if len(toks) > NOISE_MAX_TOKENS:
        return True
    alpha = sum(ch.isalpha() for ch in s)
    if alpha < NOISE_MIN_ALPHA:
        return True
    # skip very generic scaffolding
    if s.lower().startswith(("includes a ", "assess ", "determine ", "evaluate ")):
        return True
    return False

# ============================== scispaCy (optional) ==========================

_SPACY_READY = False
_SPACY_NLP = None
_SPACY_LINKER = None
_SPACY_PIPE_NAME = None

def _lazy_init_scispacy() -> None:
    global _SPACY_READY, _SPACY_NLP, _SPACY_LINKER, _SPACY_PIPE_NAME
    if _SPACY_READY:
        return
    try:
        import spacy
        _SPACY_NLP = spacy.load("en_core_sci_scibert")
        try:
            _SPACY_NLP.add_pipe(
                "scispacy_linker",
                config={"resolve_abbreviations": True, "linker_name": "umls"},
                last=True,
            )
            _SPACY_PIPE_NAME = "scispacy_linker"
        except Exception:
            try:
                from scispacy.linking import EntityLinker  # type: ignore
                from spacy.language import Language
                @Language.factory("umls_linker")
                def _create_umls_linker(nlp, name):
                    return EntityLinker(resolve_abbreviations=True, name="umls")
                _SPACY_NLP.add_pipe("umls_linker", last=True)
                _SPACY_PIPE_NAME = "umls_linker"
            except Exception:
                _SPACY_PIPE_NAME = None
        _SPACY_LINKER = _SPACY_NLP.get_pipe(_SPACY_PIPE_NAME) if _SPACY_PIPE_NAME else None
    except Exception as e:
        log.debug("[linker] scispaCy unavailable: %s", e)
        _SPACY_NLP = None
        _SPACY_LINKER = None
        _SPACY_PIPE_NAME = None
    finally:
        _SPACY_READY = True

def _scispacy_top(surface: str) -> Tuple[Optional[str], Optional[float], List[str], List[str]]:
    if not surface.strip():
        return None, None, [], []
    if not _SPACY_READY:
        _lazy_init_scispacy()
    if not _SPACY_LINKER or not _SPACY_NLP:
        return None, None, [], []
    try:
        doc = _SPACY_NLP.make_doc(surface)
        sp = doc.char_span(0, len(surface), label="ENTITY", alignment_mode="expand")
        if sp is None:
            return None, None, [], []
        doc.set_ents([sp])
        _SPACY_LINKER(doc)
        ent = doc.ents[0] if doc.ents else None
        kb_ents = list(getattr(ent._, "kb_ents", []) or []) if ent is not None else []
        if not kb_ents:
            return None, None, [], []
        cui, score = kb_ents[0]
        kb_ent = _SPACY_LINKER.kb.cui_to_entity.get(cui)
        stypes = list(getattr(kb_ent, "types", []) or []) if kb_ent else []
        srcs = list(getattr(kb_ent, "sources", []) or []) if kb_ent else []
        return str(cui), float(score or 0.0), stypes, srcs
    except Exception:
        return None, None, [], []

# ============================== Normalization ================================

def _mk_record(
    surface: str,
    cui: Optional[str],
    canonical: Optional[str],
    semantic_types: List[str],
    kb_sources: List[str],
    api_score: Optional[float],
    link_score: Optional[float],
) -> Dict[str, Any]:
    a = float(api_score or 0.0)
    l = float(link_score or 0.0)
    conf = max(0.0, min(1.0, 0.7 * a + 0.3 * l))  # blend
    return {
        "text": surface,
        "cui": cui,
        "canonical": canonical or surface,
        "semantic_types": list(semantic_types or []),
        "kb_sources": list(kb_sources or []),
        "valid": bool(cui),
        "scores": {"api": api_score, "link": link_score, "confidence": conf},
    }

def _semtypes_from_item(item: Dict[str, Any]) -> List[str]:
    sts = item.get("semanticTypes") or []
    out: List[str] = []
    for st in sts:
        name = st.get("name") or st.get("Name")
        if name:
            out.append(name)
    return out

# ============================== Variants & n-grams ============================

_PARENS_CONTENT_RE = re.compile(r"\(([^)]+)\)")
_PUNCT_SEP_RE = re.compile(r"[,/;]+")

def _generate_variants(surface: str) -> List[str]:
    s = _normalize_ws(surface)
    if not s:
        return []
    variants: List[str] = [s, s.lower()]
    title = s.title()
    if title not in variants:
        variants.append(title)

    # de-parenthesized / inside-parens
    no_paren = _PARENS_CONTENT_RE.sub("", s).strip()
    if no_paren and no_paren != s:
        variants.append(_normalize_ws(no_paren))
    for c in _PARENS_CONTENT_RE.findall(s):
        cc = _normalize_ws(c)
        if cc and cc not in variants:
            variants.append(cc)

    # hyphen/slash normalization
    s_h = s.replace("-", " ")
    if s_h != s:
        variants.append(_normalize_ws(s_h))
    parts = [p.strip() for p in _PUNCT_SEP_RE.split(s) if p.strip()]
    for p in parts:
        if p and p not in variants:
            variants.append(p)

    uniq, out = set(), []
    for v in variants:
        k = v.lower()
        if v and k not in uniq:
            uniq.add(k); out.append(v)
        if len(out) >= MAX_VARIANTS_PER_SURFACE:
            break
    return out

def _ngram_variants(surface: str) -> List[str]:
    s = _normalize_ws(surface)
    toks = _tokenize_for_ngrams(s)
    if len(toks) < MIN_NGRAM_LEN:
        return []
    ngrams: List[str] = []
    for n in range(MIN_NGRAM_LEN, min(MAX_NGRAM_LEN, len(toks)) + 1):
        for i in range(0, len(toks) - n + 1):
            phrase = " ".join(toks[i:i+n])
            if phrase and not _is_trivial(phrase):
                ngrams.append(phrase)
            if len(ngrams) >= MAX_NGRAMS_PER_SURFACE:
                break
        if len(ngrams) >= MAX_NGRAMS_PER_SURFACE:
            break
    seen, out = set(), []
    for p in ngrams:
        k = p.lower()
        if k not in seen:
            seen.add(k); out.append(p)
    return out

# ============================== UMLS /search (cached) ========================

@lru_cache(maxsize=8192)
def _cached_search(
    apikey: str,
    term: str,
    page_size: int,
    sabs: str,
    ttys: str,
    search_type: str,  # "", "words", "exact", "approximate"
) -> Tuple[
    Tuple[str, ...],
    Tuple[float, ...],
    Tuple[str, ...],
    Tuple[Tuple[str, ...], ...],
    Tuple[Tuple[str, ...], ...]
]:
    term_norm = (term or "").strip()
    if _is_trivial(term_norm) or _looks_like_noise(term_norm):
        return (), (), (), (), ()

    cache_key = f"{term_norm.lower()}|{page_size}|{sabs}|{ttys}|{search_type}"
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            try:
                v = _CACHE[cache_key]
                return (
                    tuple(v["cuis"]),
                    tuple(v["scores"]),
                    tuple(v["canon"]),
                    tuple(tuple(x) for x in v["semtypes"]),
                    tuple(tuple(x) for x in v["sources"]),
                )
            except Exception:
                pass

    params: Dict[str, Any] = {"string": term_norm, "pageSize": page_size}
    if sabs: params["sabs"] = sabs
    if ttys: params["termType"] = ttys
    if search_type: params["searchType"] = search_type  # words|exact|approximate

    try:
        data = _api_get("/search/current", _inject_key(params, apikey)).json()
        results = data.get("result", {}).get("results", []) or []
    except Exception as e:
        log.debug("[UMLS] search failed for %r (%s): %s", term_norm, search_type or "default", e)
        results = []

    cuis: List[str] = []
    api_scores: List[float] = []
    canonicals: List[str] = []
    semtypes_list: List[Tuple[str, ...]] = []
    sources_list: List[Tuple[str, ...]] = []

    for i, r in enumerate(results):
        cui = r.get("ui")
        if not cui or cui == "NONE":
            continue
        root_src = r.get("rootSource")
        cuis.append(str(cui))
        canonicals.append(r.get("name") or term_norm)
        semtypes_list.append(tuple(_semtypes_from_item(r)))
        srcs = [root_src] if root_src else []
        sources_list.append(tuple([s for s in srcs if s]))
        # UMLS doesn't provide a probability; use a simple rank-decay
        api_scores.append(max(0.3, 1.0 - 0.12 * i))

    with _CACHE_LOCK:
        _CACHE[cache_key] = {
            "cuis": cuis, "scores": api_scores, "canon": canonicals,
            "semtypes": [list(x) for x in semtypes_list],
            "sources": [list(x) for x in sources_list],
        }
        if not CACHE_DISABLE and (len(_CACHE) % CACHE_SAVE_INTERVAL == 0):
            _save_cache()

    return tuple(cuis), tuple(api_scores), tuple(canonicals), tuple(semtypes_list), tuple(sources_list)

def _umls_search_multi(
    apikey: str,
    term: str,
    page_size: int,
    sabs: str,
    ttys: str,
) -> List[Dict[str, Any]]:
    """
    Try default → words → approximate, aggregate unique CUIs (keep best score).
    """
    if _looks_like_noise(term):
        return []
    all_rows: Dict[str, Dict[str, Any]] = {}
    for mode in ("", "words"):  # skip "approximate" for speed
        try:
            cuis, api_scores, canonicals, semtypes_list, sources_list = _cached_search(
                apikey, term, page_size, sabs, ttys, mode
            )
        except Exception:
            continue
        for cui, a, can, sts, srcs in zip(cuis, api_scores, canonicals, semtypes_list, sources_list):
            kb_sources = sorted(set(list(srcs) + ["UMLS"]))
            rec = _mk_record(
                surface=term, cui=cui, canonical=can,
                semantic_types=list(sts), kb_sources=kb_sources,
                api_score=float(a), link_score=None,
            )
            prev = all_rows.get(cui)
            if not prev or (rec["scores"]["api"] or 0.0) > (prev["scores"]["api"] or 0.0):
                all_rows[cui] = rec
    return list(all_rows.values())

def umls_search(
    apikey: str,
    term: str,
    page_size: int = 7,
    *,
    sabs: Any = DEFAULT_SABS,
    ttys: Any = DEFAULT_TTYS
) -> List[Dict[str, Any]]:
    term = _normalize_ws(term)
    if not term or _is_trivial(term) or _looks_like_noise(term):
        return []

    def _canon_csv(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, (list, tuple, set)):
            vals = [str(v).strip() for v in x if str(v).strip()]
            return ",".join(vals)
        return str(x).strip()

    sabs_str = _canon_csv(sabs)
    ttys_str = _canon_csv(ttys)

    try:
        ps = int(page_size) if not isinstance(page_size, (list, tuple, set)) else int(next(iter(page_size)))
    except Exception:
        ps = 7

    try:
        return _umls_search_multi(apikey, term, int(ps), sabs_str, ttys_str)
    except Exception as e:
        log.debug("[UMLS] search failed for %r: %s", term, e)
        return []

# ============================== scispaCy reconciliation =======================

def _reconcile_with_scispacy(surface: str, candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return
    cui_local, score_local, stypes_local, srcs_local = _scispacy_top(surface)
    if cui_local is None:
        for c in candidates:
            a = float(c["scores"].get("api") or 0.0)
            c["scores"]["link"] = 0.0
            c["scores"]["confidence"] = max(0.0, min(1.0, 0.7 * a))
        return

    for c in candidates:
        if c.get("cui") == cui_local:
            c["scores"]["link"] = float(score_local or 0.0)
            if not c.get("semantic_types"):
                c["semantic_types"] = list(stypes_local or [])
            if srcs_local:
                srcs = set((c.get("kb_sources") or []) + list(srcs_local) + ["UMLS"])
                c["kb_sources"] = list(srcs)
        else:
            c["scores"]["link"] = 0.1
        a = float(c["scores"].get("api") or 0.0)
        l = float(c["scores"].get("link") or 0.0)
        c["scores"]["confidence"] = max(0.0, min(1.0, 0.7 * a + 0.3 * l))

# ============================== Public APIs ==================================

def _aggregate_candidates(surface: str, cands_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    best_by_cui: Dict[str, Dict[str, Any]] = {}
    for lst in cands_lists:
        for c in lst or []:
            cui = c.get("cui")
            if not cui:
                continue
            prev = best_by_cui.get(cui)
            if not prev or (c["scores"]["confidence"] or 0.0) > (prev["scores"]["confidence"] or 0.0):
                cc = dict(c)
                cc["text"] = surface  # keep original
                best_by_cui[cui] = cc
    merged = list(best_by_cui.values())
    merged.sort(key=lambda r: float((r.get("scores") or {}).get("confidence") or 0.0), reverse=True)
    return merged

def _link_one_surface(
    apikey: str,
    surface: str,
    top_k: int,
    scispacy_when: str,
    allowed_kb_sources: Optional[List[str]],
) -> List[Dict[str, Any]]:
    s = _normalize_ws(surface)
    if not s or _is_trivial(s) or _looks_like_noise(s):
        return []

    # 1) main
    cands_main = umls_search(apikey, s, page_size=max(7, top_k + 2))
    cands_all: List[List[Dict[str, Any]]] = [cands_main]

    # 2) variants
    for v in _generate_variants(s):
        if v == s or _looks_like_noise(v):
            continue
        cv = umls_search(apikey, v, page_size=max(7, top_k + 2))
        if cv:
            cands_all.append(cv)

    # 3) n-grams
    for ng in _ngram_variants(s):
        if _looks_like_noise(ng):
            continue
        cands_ng = umls_search(apikey, ng, page_size=max(7, top_k + 2))
        if cands_ng:
            cands_all.append(cands_ng)

    merged = _aggregate_candidates(s, cands_all)

    # Optional scispaCy reconciliation
    need_scispacy = False
    if scispacy_when == "always":
        need_scispacy = True
    elif scispacy_when == "auto" and merged:
        top_api = float((merged[0].get("scores") or {}).get("api") or 0.0)
        margin = top_api - (float((merged[1].get("scores") or {}).get("api") or 0.0) if len(merged) > 1 else 0.0)
        need_scispacy = (top_api < 0.9) or (margin < 0.12)
    if need_scispacy:
        _reconcile_with_scispacy(s, merged)

    if allowed_kb_sources:
        allow = {x.upper() for x in allowed_kb_sources}
        for c in merged:
            srcs = {y.upper() for y in (c.get("kb_sources") or [])}
            if not (srcs & allow):
                c["valid"] = False

    return merged[: max(1, int(top_k))]

def link_texts(
    *args,
    top_k: int = 3,
    scispacy_when: str = "auto",
    allowed_kb_sources: Optional[List[str]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Back-compat:
      1) link_texts(texts, *, ...)
      2) link_texts(apikey, texts, *, ...)
    Returns: List[List[Candidate]] aligned with input order.
    """
    if not args:
        raise TypeError("link_texts() missing required argument: texts")
    if isinstance(args[0], str):
        if len(args) < 2:
            raise TypeError("link_texts() missing required argument: texts")
        apikey = _ensure_key(args[0])
        texts = list(args[1])
    else:
        apikey = _ensure_key(None)
        texts = list(args[0])

    out: List[List[Dict[str, Any]]] = []
    for surface in texts or []:
        try:
            lst = _link_one_surface(
                apikey=apikey,
                surface=surface,
                top_k=top_k,
                scispacy_when=scispacy_when,
                allowed_kb_sources=allowed_kb_sources,
            )
        except Exception as e:
            log.debug("[link_texts] error linking %r: %s", surface, e)
            lst = []
        out.append(lst)
    return out

def link_texts_batch(
    surfaces: List[str],
    *,
    top_k: int = 3,
    scispacy_when: str = "auto",
    allowed_kb_sources: Optional[List[str]] = None,
    parallel: bool = False,  # NEW: supported kwarg (ignored by callers if False)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Canonical batch API (dict keyed by original surface).
    With parallel=True we use a bounded threadpool; requests are still capped by _INFLIGHT.
    """
    apikey = _ensure_key(None)
    out: Dict[str, List[Dict[str, Any]]] = {}

    # Pre-dedupe and skip noise to save calls
    seen, uniq = set(), []
    for s in surfaces or []:
        ss = _normalize_ws(s)
        if not ss or _is_trivial(ss) or ss.lower() in seen:
            continue
        seen.add(ss.lower())
        uniq.append(ss)

    if not uniq:
        return {s: [] for s in (surfaces or [])}

    def _work(surface: str) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            return surface, _link_one_surface(
                apikey=apikey,
                surface=surface,
                top_k=top_k,
                scispacy_when=scispacy_when,
                allowed_kb_sources=allowed_kb_sources,
            )
        except Exception as e:
            log.debug("[link_texts_batch] error linking %r: %s", surface, e)
            return surface, []

    if parallel:
        # threads limited by MAX_INFLIGHT to stay gentle to UMLS
        workers = min(MAX_INFLIGHT, max(2, min(8, len(uniq))))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, s): s for s in uniq}
            for fut in as_completed(futs):
                s, lst = fut.result()
                out[s] = lst
    else:
        for s in uniq:
            _, lst = _work(s)
            out[s] = lst

    # Return entries for every original surface (including skipped/noise)
    return {s: out.get(_normalize_ws(s), []) for s in (surfaces or [])}

def link_spans(
    *args,
    top_k: int = 2,
    scispacy_when: str = "auto",
    allowed_kb_sources: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Link specific spans (start, end, label) from a larger text.
    Back-compat:
      1) link_spans(text, spans, *, ...)
      2) link_spans(apikey, text, spans, *, ...)
    """
    if not args or (len(args) == 1 and not isinstance(args[0], str)):
        apikey = _ensure_key(None)
        text = args[0]
        spans = args[1]
    else:
        apikey = _ensure_key(args[0]) if isinstance(args[0], str) else _ensure_key(None)
        text = args[1] if len(args) > 1 else ""
        spans = args[2] if len(args) > 2 else []

    uniq: List[str] = []
    seen = set()
    surfaces: List[str] = []
    for (s, e, lbl) in spans or []:
        s, e = int(s), int(e)
        if e <= s:
            surfaces.append("")
            continue
        surface = (text or "")[s:e].strip()
        surfaces.append(surface)
        lk = surface.lower()
        if surface and not _is_trivial(surface) and not _looks_like_noise(surface) and not lk.startswith(("final answer", "final:", "answer:")) and lk not in seen:
            seen.add(lk); uniq.append(surface)

    by_surface: Dict[str, List[Dict[str, Any]]] = {}
    for u in uniq:
        by_surface[u.lower()] = _link_one_surface(
            apikey=apikey, surface=u, top_k=top_k,
            scispacy_when=scispacy_when, allowed_kb_sources=allowed_kb_sources
        )

    out: List[Dict[str, Any]] = []
    for (s, e, lbl), surface in zip(spans or [], surfaces):
        if not surface or _is_trivial(surface) or _looks_like_noise(surface) or surface.lower().startswith(("final answer", "final:", "answer:")):
            rec = _mk_record(surface, None, None, [], [], 0.0, 0.0)
            rec.update({"start": s, "end": e, "label": lbl, "valid": False})
            out.append(rec); continue
        cands = list(by_surface.get(surface.lower()) or [])
        best = dict(cands[0]) if cands else _mk_record(surface, None, None, [], [], 0.0, 0.0)
        if not cands:
            best["valid"] = False
        best["start"] = int(s); best["end"] = int(e); best["label"] = lbl
        out.append(best)
    return out

# --- Compatibility shims / extra tools ---------------------------------------

def is_configured_legacy() -> bool:
    try:
        from config import UMLS_API_KEY as _K
    except Exception:
        _K = os.getenv("UMLS_API_KEY", "")
    return bool(_K)

def link_terms(terms, top_k=5, sabs=None, search_type="words"):
    """
    Legacy interface:
    returns [{"surface": <text>, "candidates": [{cui,name,sab,uri,score}...]}...]
    """
    results = link_texts(terms, top_k=top_k)
    out = []
    for surface, cand_list in zip(terms, results):
        mapped = []
        for c in cand_list:
            cui = c.get("cui")
            name = c.get("canonical") or c.get("text") or ""
            kb_sources = c.get("kb_sources") or []
            sab = (kb_sources[0] if kb_sources else "") or "UMLS"
            score = (c.get("scores") or {}).get("confidence")
            uri = f"{UTS_REST}/content/current/CUI/{cui}" if cui else ""
            mapped.append({"cui": cui, "name": name, "sab": sab, "uri": uri, "score": score})
        out.append({"surface": surface, "candidates": mapped[: max(1, int(top_k))]})
    return out

__all__ = list(set([
    *globals().get("__all__", []),
    # public
    "link_texts", "link_texts_batch", "link_spans", "umls_search",
    # extra helpers / tools
    "search_strings", "codes_to_cuis", "cuis_to_codes", "crosswalk",
    "content_by_cui_cached", "atoms_for_cui", "hierarchy_for_source_code",
    # shims
    "is_configured", "is_configured_legacy", "link_terms",
]))

# ============================== Tooling APIs (optional) ======================

def _load_list(inline: List[str], inputfile: Optional[str]) -> List[str]:
    if inputfile:
        with open(inputfile, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    else:
        lines = [s.strip() for s in inline if str(s).strip()]
    seen, out = set(), []
    for x in lines:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def search_strings(
    apikey: str,
    version: str,
    terms: Iterable[str],
    sabs: Optional[str],
    ttys: Optional[str],
    page_size: int,
    max_pages: int,
    search_type: Optional[str] = None,
    include_obsolete: Optional[bool] = None,
    include_suppressible: Optional[bool] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    for term in terms:
        if _is_trivial(term) or _looks_like_noise(term):
            results[term] = []; continue
        acc: List[Dict[str, Any]] = []; page = 0
        while True:
            page += 1
            params: Dict[str, Any] = {"string": term, "pageNumber": page, "pageSize": page_size}
            if sabs: params["sabs"] = sabs
            if ttys: params["termType"] = ttys
            if search_type: params["searchType"] = search_type
            if include_obsolete is not None: params["includeObsolete"] = str(include_obsolete).lower()
            if include_suppressible is not None: params["includeSuppressible"] = str(include_suppressible).lower()
            r = _api_get(f"/search/{version}", _inject_key(params, apikey))
            data = r.json(); rows = (data.get("result") or {}).get("results") or []
            if not rows: break
            acc.extend(rows)
            if page >= max_pages: break
        results[term] = acc
    return results

def codes_to_cuis(apikey: str, version: str, source: str, codes: Iterable[str], page_size: int, max_pages: int) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for code in codes:
        acc: List[Dict[str, Any]] = []
        if _is_trivial(code) or _looks_like_noise(code):
            out[code] = []; continue
        page = 0
        while True:
            page += 1
            params = {"string": code, "rootSource": source, "inputType": "sourceUI", "pageNumber": page, "pageSize": page_size}
            r = _api_get(f"/search/{version}", _inject_key(params, apikey))
            data = r.json(); rows = (data.get("result") or {}).get("results") or []
            if not rows: break
            acc.extend(rows)
            if page >= max_pages: break
        out[code] = acc
    return out

def cuis_to_codes(apikey: str, version: str, sabs: str, cuis: Iterable[str], page_size: int, max_pages: int) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cui in cuis:
        acc: List[Dict[str, Any]] = []; page = 0
        while True:
            page += 1
            params = {"string": cui, "sabs": sabs, "returnIdType": "code", "pageNumber": page, "pageSize": page_size}
            r = _api_get(f"/search/{version}", _inject_key(params, apikey))
            data = r.json(); rows = (data.get("result") or {}).get("results") or []
            if not rows: break
            acc.extend(rows)
            if page >= max_pages: break
        out[cui] = acc
    return out

def crosswalk(apikey: str, version: str, source: str, target: str, codes: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for code in codes:
        params = {"targetSource": target}
        r = _api_get(f"/crosswalk/{version}/source/{source}/{code}", _inject_key(params, apikey))
        data = r.json(); rows = (data.get("result") or [])
        out[code] = rows
    return out

@lru_cache(maxsize=4096)
def content_by_cui_cached(apikey: str, version: str, cui: str) -> Dict[str, Any]:
    r = _api_get(f"/content/{version}/CUI/{cui}", _inject_key({}, apikey))
    return r.json().get("result") or {}

def atoms_for_cui(apikey: str, version: str, cui: str, page_size: int = 25, max_pages: int = 100) -> List[Dict[str, Any]]:
    root = content_by_cui_cached(apikey, version, cui)
    atoms_url = root.get("atoms")
    if not atoms_url: return []
    acc: List[Dict[str, Any]] = []; page = 0
    while True:
        page += 1
        rr = SES.get(atoms_url, params={"apiKey": apikey, "pageNumber": page, "pageSize": page_size}, timeout=(CONNECT_TIMEOUT, REQ_TIMEOUT))
        if rr.status_code != 200: break
        data = rr.json(); items = data.get("result") or []
        if not items: break
        acc.extend(items)
        if page >= max_pages: break
    return acc

def hierarchy_for_source_code(apikey: str, version: str, source: str, identifier: str, operation: str, page_size: int = 25, max_pages: int = 100) -> List[Dict[str, Any]]:
    acc: List[Dict[str, Any]] = []; page = 0
    while True:
        page += 1
        params = {"pageNumber": page, "pageSize": page_size}
        r = _api_get(f"/content/{version}/source/{source}/{identifier}/{operation}", _inject_key(params, apikey))
        items = r.json().get("result") or []
        if not items: break
        acc.extend(items)
        if page >= max_pages: break
    return acc

# ============================== CLI plumbing =================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UMLS Toolkit (apiKey mode — robust pool/parallel)")
    default_key = UMLS_API_KEY or os.getenv("UMLS_API_KEY", "") or None
    p.add_argument("--apikey", "-k", default=default_key, help="UMLS API key (env/config fallback)")
    p.add_argument("--version", "-v", default=DEFAULT_VERSION, help="UMLS version (default: current)")
    p.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"], help="Log level")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("link-texts", help="Batch link terms → UMLS candidates (variants + ngrams).")
    sp.add_argument("--terms", nargs="*", default=[])
    sp.add_argument("--input", help="File with one term per line")
    sp.add_argument("--top-k", type=int, default=3)
    sp.add_argument("--scispacy-when", choices=["auto", "always", "never"], default="auto")
    sp.add_argument("--allow", help="Allow-list sources (comma-separated), e.g. 'MSH,SNOMEDCT_US,RXNORM'")
    sp.add_argument("--parallel", action="store_true")
    sp.add_argument("--json", action="store_true")

    return p

def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log))

    apikey = _ensure_key(args.apikey)
    if args.cmd == "link-texts":
        terms = _load_list(args.terms or [], args.input)
        allow = [s.strip() for s in (args.allow or "").split(",") if s.strip()] if args.allow else None
        res = link_texts_batch(terms, top_k=args.top_k, scispacy_when=args.scispacy_when, allowed_kb_sources=allow, parallel=args.parallel)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False)); return 0
        for t in terms:
            print(f"\n=== {t} ===")
            for r in res.get(t, []):
                print({
                    "text": r["text"],
                    "cui": r["cui"],
                    "canonical": r["canonical"],
                    "semantic_types": r["semantic_types"],
                    "kb_sources": r["kb_sources"],
                    "scores": r["scores"],
                    "valid": r["valid"],
                })
        return 0

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
