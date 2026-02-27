#!/usr/bin/env python3
# utils/hybrid_checker.py
# -*- coding: utf-8 -*-
"""
Hybrid entailment checker for biomedical CoT steps.

What this provides
------------------
• build_entailment_records(steps, umls_per_step, ...) → list of dicts per adjacent pair:
  { "step_pair":[i, j], "probs":{"entailment":..., "neutral":..., "contradiction":...}, "final_label":"entailment" }

• Fast, batched inference with Transformers when available:
  - Default model candidates (in order):
      1) $BIO_NLI_MODEL
      2) "Bam3752/PubMedBERT-BioNLI-LoRA"  (specialized BioNLI)
      3) "roberta-large-mnli"              (general NLI fallback)
  - Uses CUDA, then MPS (Apple Silicon), else CPU.
  - No-transformers fallback: heuristic-only scores (still returns non-"unknown" labels).

• UMLS-aware adjustments:
  - Jaccard overlap over CUIs boosts entailment.
  - Strong negation pattern mismatch boosts contradiction.
  - Light normalization & renormalization to keep probabilities valid.

• Safe imports and graceful degradation (won’t crash your run).
"""

from __future__ import annotations

import math
import os
import re
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("hybrid_nli")

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

_NEG_RE = re.compile(r"\b(no|not|never|without|den(y|ies|ied)|absence|rule\s*out|contraindicat(ed|ion)|prevents?)\b", re.IGNORECASE)
_POS_RE = re.compile(r"\b(increase(s|d)?|raise(s|d)?|cause(s|d)?|lead(s|ing)?\s*to|result(s|ed)?\s*in|associated\s*with)\b", re.IGNORECASE)
_ANT_RE = re.compile(r"\b(decrease(s|d)?|reduce(s|d)?|mitigate(s|d)?|prevent(s|ed)?|protect(s|ive)?)\b", re.IGNORECASE)

def _softmax(xs: Iterable[float]) -> List[float]:
    xs = list(xs)
    if not xs:
        return [1.0, 0.0, 0.0]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _renorm(probs: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in probs.values()) or 1.0
    return {k: max(0.0, v) / s for k, v in probs.items()}

def _neg_present(text: str) -> bool:
    return bool(_NEG_RE.search(text or ""))

def _pos_present(text: str) -> bool:
    return bool(_POS_RE.search(text or ""))

def _ant_present(text: str) -> bool:
    return bool(_ANT_RE.search(text or ""))

def _collect_cuis(step_umls: List[Dict[str, Any]]) -> List[str]:
    cuis = []
    for c in step_umls or []:
        cui = str((c or {}).get("cui") or "").strip()
        if cui:
            cuis.append(cui.upper())
    # dedupe preserve order
    seen, out = set(), []
    for c in cuis:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

_TOK = None
_MOD = None
_DEV = "cpu"
_ID2LABEL: Dict[int, str] = {}
_LABEL2IDX: Dict[str, int] = {}
_MODEL_NAME = None
_TRANSFORMERS_OK = False

def _pick_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

def _try_load(model_name: str) -> bool:
    global _TOK, _MOD, _DEV, _ID2LABEL, _LABEL2IDX, _MODEL_NAME, _TRANSFORMERS_OK
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
        import torch  # type: ignore

        _DEV = _pick_device()
        log.info("[entailment] loading NLI model: %s (device=%s)", model_name, _DEV)
        _TOK = AutoTokenizer.from_pretrained(model_name)
        _MOD = AutoModelForSequenceClassification.from_pretrained(model_name)
        _MOD.eval()
        if _DEV != "cpu":
            _MOD.to(_DEV)

        id2label = getattr(_MOD.config, "id2label", None) or {}
        _ID2LABEL = {int(k): str(v) for k, v in (id2label.items() if isinstance(id2label, dict) else [])}
        if not _ID2LABEL:
            # common for some checkpoints
            _ID2LABEL = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

        _LABEL2IDX = {v.upper(): k for k, v in _ID2LABEL.items()}
        # Normalize a few variants
        for k, v in list(_LABEL2IDX.items()):
            if k.startswith("LABEL_"):
                # assume 0=contradiction, 1=neutral, 2=entailment
                _LABEL2IDX = {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}
                _ID2LABEL = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
                break

        _MODEL_NAME = model_name
        _TRANSFORMERS_OK = True
        return True
    except Exception as e:
        log.info("[entailment] could not load %s: %s", model_name, e)
        _TRANSFORMERS_OK = False
        return False

def _ensure_model() -> bool:
    # Set FORCE_HEURISTIC_NLI=1 to skip all model loading and use the fast heuristic
    if os.getenv("FORCE_HEURISTIC_NLI", "").strip() in ("1", "true", "yes"):
        return False
    if _TRANSFORMERS_OK and _TOK is not None and _MOD is not None:
        return True
    # try candidates in order
    for name in [
        os.getenv("BIO_NLI_MODEL", "").strip() or "",
        "Bam3752/PubMedBERT-BioNLI-LoRA",
        "roberta-large-mnli",
    ]:
        if name and _try_load(name):
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# NLI scoring
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_nli(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Lightweight heuristic NLI based on token overlap and biomedical negation/direction patterns.
    Used as a fallback when no transformer model is available.

    Signals:
      • Token overlap → entailment prior  (consecutive CoT steps share vocabulary)
      • Negation mismatch (neg_i XOR neg_j) → contradiction boost
      • Opposite-direction verbs (increase vs decrease) → contradiction boost
    """
    p_toks = set(re.sub(r"[^a-z0-9 ]", " ", (premise or "").lower()).split())
    h_toks = set(re.sub(r"[^a-z0-9 ]", " ", (hypothesis or "").lower()).split())

    # Remove stopwords
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
              "being", "have", "has", "had", "do", "does", "did", "will",
              "would", "could", "should", "may", "might", "in", "of", "to",
              "and", "or", "but", "that", "this", "it", "its", "at", "by",
              "for", "with", "as", "on", "from", "can"}
    p_toks -= _STOP
    h_toks -= _STOP

    # Token overlap ratio — use Jaccard (symmetric) rather than precision to
    # avoid over-scoring when one step is a subset of the other.
    inter = len(p_toks & h_toks)
    union = len(p_toks | h_toks) or 1
    overlap = inter / union

    neg_p = _neg_present(premise)
    neg_h = _neg_present(hypothesis)
    pos_p = _pos_present(premise)
    pos_h = _pos_present(hypothesis)
    ant_p = _ant_present(premise)
    ant_h = _ant_present(hypothesis)

    # --- contradiction signals ---
    contra = 0.08  # modest base — some step pairs DO contradict
    if neg_p ^ neg_h:                        # one negated, the other not
        contra += 0.35
    if (pos_p and ant_h) or (ant_p and pos_h):  # opposite effect verbs
        contra += 0.25

    # --- entailment signals ---
    # Cap entailment lower — consecutive steps share vocabulary by design,
    # so overlap alone is weak evidence of logical entailment.
    entail = min(0.50, overlap * 0.60)

    # --- neutral as default ---
    # Lean more neutral: without strong directional signals, steps are
    # often merely topically related, not logically entailing.
    neutral = max(0.15, 1.0 - entail - contra)

    return _renorm({"entailment": entail, "neutral": neutral, "contradiction": contra})


def _nli_scores_batch(pairs: List[Tuple[str, str]], max_length: int = 256, batch_size: int = 8) -> List[Dict[str, float]]:
    """
    Returns a list of dicts with keys: entailment, neutral, contradiction (probabilities).
    Falls back to token-overlap heuristic if model not present.
    """
    if not _ensure_model():
        return [_heuristic_nli(p, h) for p, h in pairs]

    from math import ceil
    import torch  # type: ignore

    outs: List[Dict[str, float]] = []
    n = len(pairs)
    if n == 0:
        return outs

    for b in range(0, n, batch_size):
        chunk = pairs[b : b + batch_size]
        p_texts = [p for p, _ in chunk]
        h_texts = [h for _, h in chunk]
        enc = _TOK(
            p_texts,
            h_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if _DEV != "cpu":
            enc = {k: v.to(_DEV) for k, v in enc.items()}
        with torch.no_grad():
            logits = _MOD(**enc).logits  # [B, C]
        if _DEV != "cpu":
            logits = logits.detach().cpu()
        for row in logits:
            row = row.tolist()
            # Map to [C, N, E] indices
            try:
                c_idx = _LABEL2IDX.get("CONTRADICTION", 0)
                n_idx = _LABEL2IDX.get("NEUTRAL", 1)
                e_idx = _LABEL2IDX.get("ENTAILMENT", 2)
            except Exception:
                c_idx, n_idx, e_idx = 0, 1, 2
            vec = [row[e_idx], row[n_idx], row[c_idx]]  # we'll softmax ourselves for robustness
            probs = _softmax(vec)
            outs.append({"entailment": probs[0], "neutral": probs[1], "contradiction": probs[2]})
    return outs

# ─────────────────────────────────────────────────────────────────────────────
# UMLS-aware adjustments
# ─────────────────────────────────────────────────────────────────────────────

def _adjust_with_umls(
    base_probs: Dict[str, float],
    step_i: str,
    step_j: str,
    umls_i: List[Dict[str, Any]],
    umls_j: List[Dict[str, Any]],
    relation_info: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """
    Adjustments based on UMLS overlap, semantic type relations, and negation cues.

    Design: adjustments are balanced so the base NLI model's decision is nudged,
    not overridden. CUI overlap boosts are intentionally small because consecutive
    CoT steps naturally share medical concepts.
    """
    probs = dict(base_probs)

    # 1) CUI overlap — light adjustments only
    cuis_i = _collect_cuis(umls_i)
    cuis_j = _collect_cuis(umls_j)
    jac = _jaccard(cuis_i, cuis_j)
    if jac >= 0.5:
        probs["entailment"] += 0.03
        probs["neutral"] -= 0.02
    elif jac >= 0.25:
        probs["entailment"] += 0.02
        probs["neutral"] -= 0.01
    elif (cuis_i and cuis_j) and jac == 0.0:
        # Disjoint concept sets suggest a topic shift
        probs["neutral"] += 0.04
        probs["contradiction"] += 0.03

    # 2) Relation-aware adjustments (from umls_checker)
    if relation_info:
        has_any_allowed = any(r.get("allowed") for r in relation_info)
        has_any_violation = any(not r.get("allowed") for r in relation_info)

        if has_any_allowed and not has_any_violation:
            probs["entailment"] += 0.04
        elif has_any_violation and not has_any_allowed:
            probs["contradiction"] += 0.06
            probs["entailment"] -= 0.03

    # 3) Negation/antonym mismatches → boost contradiction
    neg_i, neg_j = _neg_present(step_i), _neg_present(step_j)
    pos_i, pos_j = _pos_present(step_i), _pos_present(step_j)
    ant_i, ant_j = _ant_present(step_i), _ant_present(step_j)

    has_negation_mismatch = neg_i ^ neg_j
    has_direction_mismatch = (pos_i and ant_j) or (ant_i and pos_j)

    if has_negation_mismatch or has_direction_mismatch:
        e = probs.get("entailment", 0.0)
        c = probs.get("contradiction", 0.0)
        gap = max(0.0, e - c)

        if has_negation_mismatch and has_direction_mismatch:
            boost = max(0.25, gap * 0.55)
        elif has_negation_mismatch:
            boost = max(0.18, gap * 0.45)
        else:
            boost = max(0.12, gap * 0.35)

        probs["contradiction"] += boost
        probs["entailment"] -= boost * 0.7

    probs = _renorm(probs)
    return probs

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_entailment_records(
    steps: List[str],
    umls_per_step: Optional[List[List[Dict[str, Any]]]] = None,
    *,
    max_length: int = 256,
    batch_size: int = 8,
    use_umls: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute pairwise NLI for adjacent steps with optional UMLS-informed adjustments.

    Returns a list of records like:
      {
        "step_pair": [i, j],
        "probs": {"entailment": 0.73, "neutral": 0.20, "contradiction": 0.07},
        "final_label": "entailment",
        "meta": {"model": "...", "device": "...", "overlap_jaccard": 0.33}
      }
    """
    steps = list(steps or [])
    n = len(steps)
    if n < 2:
        return []

    umls_per_step = umls_per_step or [[] for _ in range(n)]
    # protect length mismatch
    if len(umls_per_step) < n:
        umls_per_step = umls_per_step + [[] for _ in range(n - len(umls_per_step))]
    umls_per_step = umls_per_step[:n]

    # Build adjacent pairs
    pairs: List[Tuple[int, int]] = [(i, i + 1) for i in range(n - 1)]
    texts: List[Tuple[str, str]] = [(steps[i], steps[j]) for (i, j) in pairs]

    base_scores = _nli_scores_batch(texts, max_length=max_length, batch_size=batch_size)

    # Compute UMLS relation diagnostics for all adjacent pairs
    _rel_by_pair: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    has_real_concepts = any(any(c.get("cui") for c in step_umls) for step_umls in umls_per_step)
    if use_umls and has_real_concepts:
        try:
            from utils.umls_checker import make_checker, validate_relations
            checker = make_checker(enable_relation_check=True)
            rel_diags = validate_relations(umls_per_step, checker=checker)
            for rd in rel_diags:
                key = (rd["i"], rd["j"])
                _rel_by_pair.setdefault(key, []).append(rd)
        except Exception as e:
            log.debug("[hybrid] relation validation skipped: %s", e)

    out: List[Dict[str, Any]] = []
    for (i, j), base in zip(pairs, base_scores):
        probs = dict(base)
        pair_rels = _rel_by_pair.get((i, j), [])
        umls_i = umls_per_step[i] or []
        umls_j = umls_per_step[j] or []

        if use_umls:
            probs = _adjust_with_umls(
                probs, steps[i], steps[j], umls_i, umls_j,
                relation_info=pair_rels,
            )
        # final label
        label = max(probs.items(), key=lambda kv: kv[1])[0]

        # Compute relation flags for downstream guard signals
        has_relation_violation = bool(pair_rels and not any(r.get("allowed") for r in pair_rels))
        has_ontology_support = bool(pair_rels and any(r.get("allowed") for r in pair_rels))
        umls_jac = _jaccard(_collect_cuis(umls_i), _collect_cuis(umls_j))

        rec = {
            "step_pair": [i, j],
            "probs": {
                "entailment": float(probs.get("entailment", 0.0)),
                "neutral": float(probs.get("neutral", 0.0)),
                "contradiction": float(probs.get("contradiction", 0.0)),
            },
            "final_label": "entailment" if label == "entailment" else ("contradiction" if label == "contradiction" else "neutral"),
            "meta": {
                "model": _MODEL_NAME or "heuristic" if not _TRANSFORMERS_OK else (_MODEL_NAME or "unknown"),
                "device": _pick_device() if not _TRANSFORMERS_OK else _DEV,
                "umls_overlap_jaccard": umls_jac,
                "relation_violation": has_relation_violation,
                "ontology_support": has_ontology_support,
                "relation_details": pair_rels[:3] if pair_rels else [],
                "concepts_i_count": len([c for c in umls_i if c.get("valid")]),
                "concepts_j_count": len([c for c in umls_j if c.get("valid")]),
            },
        }
        out.append(rec)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Small self-test (optional)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_steps = [
        "Aspirin reduces platelet aggregation.",
        "Therefore aspirin decreases thrombus formation.",
        "Aspirin does not increase bleeding risk.",
    ]
    demo_umls = [
        [{"cui": "C0004057"}],  # Aspirin
        [{"cui": "C0040034"}],  # Thrombus
        [{"cui": "C0004057"}, {"cui": "C0019080"}],  # Aspirin, Bleeding
    ]
    recs = build_entailment_records(demo_steps, demo_umls, batch_size=2)
    print(recs)

# Public aliases for internal helpers (used by notebooks)
jaccard = _jaccard
collect_cuis = _collect_cuis
