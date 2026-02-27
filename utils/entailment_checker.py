# utils/entailment_checker.py
from __future__ import annotations

"""
Biomedical NLI (entailment) wrapper with graceful fallbacks.

Exports expected by main.py:
  - check_entailment(steps, model_name=None, batch_size=16) -> List[Dict]
  - check_entailment_bidirectional(steps, ...) -> List[Dict]
  - attach_final_labels(pairs) -> List[Dict]

Backward-compat aliases:
  - pairwise_entailment = check_entailment
  - pairwise_entailment_bidirectional = check_entailment_bidirectional

Behavior:
  * For a list of steps S0..Sn, we score adjacent pairs:
      P0 = (S0 -> S1), P1 = (S1 -> S2), ...
    Each record: {"i": i, "j": i+1, "label": "entailment|neutral|contradiction",
                  "probs": {"entailment": float, "neutral": float, "contradiction": float}}

  * If a transformer model can't be loaded (missing deps/offline), we use a
    fast heuristic (token overlap & negation) to produce non-trivial labels.

Notes:
  * Model is loaded lazily and cached once per process.
  * Works on CPU by default; uses CUDA/MPS automatically when available.
  * Robust to label orderings (maps id2label to e/n/c).
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Config / candidates
# ─────────────────────────────────────────────────────────────────────────────

# Try domain models first; fall back to strong general NLI.
DEFAULT_MODEL_CANDIDATES = [
    # Biomedical-ish (seen in your logs)
    "Bam3752/PubMedBERT-BioNLI-LoRA",
    # General-purpose strong NLI checkpoints:
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "microsoft/deberta-large-mnli",
]

ENV_MODEL = os.getenv("BIO_NLI_MODEL", "").strip()
ENV_DEVICE = os.getenv("BIO_NLI_DEVICE", "").strip()  # "", "cpu", "cuda", "mps"
ENV_BATCH = int(os.getenv("BIO_NLI_BATCH", "16"))

log = logging.getLogger("entailment")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy HF model loader (optional)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _NLIModel:
    name: str
    tokenizer: Any
    model: Any
    device: str
    id2label: Dict[int, str]
    label2id: Dict[str, int]

_NLI_SINGLETON: Optional[_NLIModel] = None
_TRY_IMPORT_FAILED = False


def _has_torch_cuda() -> bool:
    try:
        import torch  # noqa
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _has_torch_mps() -> bool:
    # macOS MPS
    try:
        import torch  # noqa
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _select_device() -> str:
    if ENV_DEVICE in {"cpu", "cuda", "mps"}:
        return ENV_DEVICE
    if _has_torch_cuda():
        return "cuda"
    if _has_torch_mps():
        return "mps"
    return "cpu"


def _normalize_label_map(id2label: Dict[int, str]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Return maps such that labels are exactly in {'entailment','neutral','contradiction'}.
    Handles cases like 'LABEL_0' or different capitalizations/orderings.
    """
    canon = {}
    for i, raw in id2label.items():
        s = str(raw).lower()
        if "entail" in s:
            canon[i] = "entailment"
        elif "contra" in s:
            canon[i] = "contradiction"
        elif "neutral" in s or s.endswith("_1") or s.endswith("label_1"):
            # crude but works when labels are generic
            canon[i] = "neutral"
        else:
            # last resort: map by common sets
            if s in {"label_0", "0", "e"}:
                canon[i] = "entailment"
            elif s in {"label_1", "1", "n"}:
                canon[i] = "neutral"
            elif s in {"label_2", "2", "c"}:
                canon[i] = "contradiction"
            else:
                # If truly unknown, assign neutral so we never crash
                canon[i] = "neutral"
    label2id = {v: k for k, v in canon.items()}
    return canon, label2id


def _load_nli_model(model_name: Optional[str] = None) -> Optional[_NLIModel]:
    global _NLI_SINGLETON, _TRY_IMPORT_FAILED
    if _NLI_SINGLETON is not None:
        return _NLI_SINGLETON
    if _TRY_IMPORT_FAILED:
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa
        import torch  # noqa
    except Exception as e:
        _TRY_IMPORT_FAILED = True
        log.info("[entailment] transformers/torch unavailable (%s) — using heuristic fallback", e)
        return None

    device = _select_device()
    cands = [model_name] if model_name else []
    if ENV_MODEL:
        cands.insert(0, ENV_MODEL)
    cands.extend([m for m in DEFAULT_MODEL_CANDIDATES if m not in cands])

    last_err: Optional[Exception] = None
    for name in cands:
        if not name:
            continue
        try:
            log.info("[entailment] loading NLI model: %s", name)
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            # move to device
            if device == "cuda":
                mdl = mdl.cuda()
            elif device == "mps":
                mdl = mdl.to("mps")
            id2label = getattr(mdl.config, "id2label", {0: "entailment", 1: "neutral", 2: "contradiction"})
            id2label_norm, label2id = _normalize_label_map(id2label)
            _NLI_SINGLETON = _NLIModel(name=name, tokenizer=tok, model=mdl, device=device,
                                       id2label=id2label_norm, label2id=label2id)
            return _NLI_SINGLETON
        except Exception as e:
            last_err = e
            log.warning("[entailment] failed to load %s: %s", name, e)

    if last_err:
        log.info("[entailment] all model candidates failed; using heuristic fallback: %s", last_err)
    _TRY_IMPORT_FAILED = True
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback (fast, no deps)
# ─────────────────────────────────────────────────────────────────────────────

_NEG_MARKERS = {" no ", " not ", " never ", " none ", " without ", " lacks ", " lack ", " absent "}


def _tokenize(t: str) -> List[str]:
    return [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in t).split() if w]


def _negated(s: str) -> bool:
    s_pad = f" {s.lower()} "
    return any(m in s_pad for m in _NEG_MARKERS)


def _heuristic_pair(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Simple lexical overlap + negation detector.
    Returns probs over E/N/C that sum to 1.
    """
    p_toks, h_toks = _tokenize(premise), _tokenize(hypothesis)
    if not p_toks or not h_toks:
        return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

    # Use Jaccard similarity (symmetric) to avoid inflated scores when
    # consecutive CoT steps share vocabulary by construction.
    p_set, h_set = set(p_toks), set(h_toks)
    inter = len(p_set & h_set)
    union = len(p_set | h_set) or 1
    sim = inter / union  # 0..1

    p_neg = _negated(premise)
    h_neg = _negated(hypothesis)

    if sim > 0.65 and p_neg == h_neg:
        e = min(0.70, 0.35 + 0.4 * sim)
        c = 0.08
        n = 1.0 - e - c
    elif sim > 0.4 and p_neg != h_neg:
        c = min(0.80, 0.35 + 0.6 * sim)
        e = 0.05
        n = 1.0 - e - c
    else:
        n = min(0.70, 0.50 + 0.3 * (1 - abs(int(p_neg) - int(h_neg))))
        rem = 1.0 - n
        e = rem * 0.4 * (1 + sim)
        c = rem - e

    # normalize just in case
    s = e + n + c
    if s <= 0:
        return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}
    return {"entailment": e / s, "neutral": n / s, "contradiction": c / s}


# ─────────────────────────────────────────────────────────────────────────────
# Core APIs
# ─────────────────────────────────────────────────────────────────────────────

def _adjacent_pairs(steps: List[str]) -> List[Tuple[int, int, str, str]]:
    pairs: List[Tuple[int, int, str, str]] = []
    for i in range(len(steps) - 1):
        a = (steps[i] or "").strip()
        b = (steps[i + 1] or "").strip()
        pairs.append((i, i + 1, a, b))
    return pairs


def _predict_hf_batch(model: _NLIModel, pairs: List[Tuple[int, int, str, str]], batch_size: int) -> List[Dict[str, Any]]:
    import torch  # type: ignore
    tok, mdl = model.tokenizer, model.model
    e_key = "entailment"; n_key = "neutral"; c_key = "contradiction"

    out: List[Dict[str, Any]] = []
    for b in range(0, len(pairs), max(1, int(batch_size))):
        chunk = pairs[b:b + batch_size]
        inputs = tok(
            [p for _, _, p, _ in chunk],
            [h for _, _, _, h in chunk],
            padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        if model.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif model.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            logits = mdl(**inputs).logits  # [B, C]
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        for (i, j, _, _), pvec in zip(chunk, probs):
            # Map logits to canonical keys via id2label
            pdict = {model.id2label[idx]: float(val) for idx, val in enumerate(pvec)}
            # Ensure all three keys exist
            e = float(pdict.get(e_key, 0.0))
            n = float(pdict.get(n_key, 0.0))
            c = float(pdict.get(c_key, 0.0))
            # Re-normalize if necessary
            s = e + n + c
            if s <= 0:
                e, n, c = 0.33, 0.34, 0.33
            else:
                e, n, c = e / s, n / s, c / s

            label = max([("entailment", e), ("neutral", n), ("contradiction", c)], key=lambda t: t[1])[0]
            out.append({"i": i, "j": j, "label": label, "probs": {"entailment": e, "neutral": n, "contradiction": c}})
    return out


def _predict_heuristic(pairs: List[Tuple[int, int, str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, j, a, b in pairs:
        probs = _heuristic_pair(a, b)
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        out.append({"i": i, "j": j, "label": label, "probs": probs})
    return out


def check_entailment(
    steps: List[str],
    model_name: Optional[str] = None,
    batch_size: int = ENV_BATCH,
) -> List[Dict[str, Any]]:
    """
    Score adjacent pairs (Si -> Si+1). Returns list of dicts as described above.
    """
    try:
        if not isinstance(steps, list) or len(steps) < 2:
            return []
        pairs = _adjacent_pairs(steps)
        mdl = _load_nli_model(model_name)
        if mdl is None:
            return _predict_heuristic(pairs)
        return _predict_hf_batch(mdl, pairs, batch_size=batch_size)
    except Exception as e:
        log.warning("[entailment] check_entailment failed (%s); using heuristic fallback", e)
        return _predict_heuristic(_adjacent_pairs(steps))


def check_entailment_bidirectional(
    steps: List[str],
    model_name: Optional[str] = None,
    batch_size: int = ENV_BATCH,
) -> List[Dict[str, Any]]:
    """
    Score both (Si -> Si+1) and (Si+1 -> Si) and merge:
      - If both entailment: 'entailment' (boosted E)
      - If one entails and the other contradicts: 'contradiction'
      - Else choose higher-confidence label from forward pass
    """
    if not isinstance(steps, list) or len(steps) < 2:
        return []

    # Forward
    fwd = check_entailment(steps, model_name=model_name, batch_size=batch_size)

    # Reverse compute by swapping each pair’s texts
    pairs = _adjacent_pairs(steps)
    rev_pairs = [(i, j, b, a) for (i, j, a, b) in pairs]
    mdl = _load_nli_model(model_name)
    if mdl:
        rev = _predict_hf_batch(mdl, rev_pairs, batch_size=batch_size)
    else:
        rev = _predict_heuristic(rev_pairs)

    out: List[Dict[str, Any]] = []
    for f, r in zip(fwd, rev):
        # Default to forward
        label = f["label"]
        pe = float(f["probs"]["entailment"]); pn = float(f["probs"]["neutral"]); pc = float(f["probs"]["contradiction"])
        re_ = float(r["probs"]["entailment"]); rn = float(r["probs"]["neutral"]); rc = float(r["probs"]["contradiction"])

        # Merge logic
        if f["label"] == "entailment" and r["label"] == "entailment":
            label = "entailment"
            pe = min(1.0, (pe + re_) / 2 + 0.1)  # light boost for symmetric entailment
            # renormalize
            s = pe + pn + pc
            pe, pn, pc = pe / s, pn / s, pc / s
        elif {"entailment", "contradiction"} == {f["label"], r["label"]}:
            label = "contradiction"
            pc = max(pc, rc)

        out.append({
            "i": f["i"], "j": f["j"], "label": label,
            "probs": {"entailment": pe, "neutral": pn, "contradiction": pc}
        })
    return out


def attach_final_labels(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure each pair has a 'label' equal to argmax(probs).
    """
    out = []
    for p in pairs or []:
        probs = p.get("probs") or {}
        if not probs:
            p["probs"] = {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
            p["label"] = "neutral"
        else:
            label = max(probs.items(), key=lambda kv: float(kv[1]))[0]
            p["label"] = label
        out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat aliases (older mains may import these)
# ─────────────────────────────────────────────────────────────────────────────

pairwise_entailment = check_entailment
pairwise_entailment_bidirectional = check_entailment_bidirectional
