#!/usr/bin/env python3
"""Streamlit GUI for Biomedical Ontology-Based Semantic Leakage Detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ─── path ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── OpenRouter catalogue ─────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_CATALOGUE: Dict[str, str] = {
    "Claude Haiku 4.5":         "anthropic/claude-haiku-4-5",
    "Claude Sonnet 4.5":        "anthropic/claude-sonnet-4-5",
    "GPT-4o Mini":              "openai/gpt-4o-mini",
    "GPT-4o":                   "openai/gpt-4o",
    "Gemini Flash 1.5":         "google/gemini-flash-1.5",
    "Gemini Pro 1.5":           "google/gemini-pro-1.5",
    "Llama 3.3 70B (free)":     "meta-llama/llama-3.3-70b-instruct:free",
    "Mistral 7B (free)":        "mistralai/mistral-7b-instruct:free",
    "DeepSeek R1 (free)":       "deepseek/deepseek-r1:free",
}
DEFAULT_SELECTED = {
    "Claude Haiku 4.5",
    "GPT-4o Mini",
    "Gemini Flash 1.5",
    "Llama 3.3 70B (free)",
}

SAMPLE_QUESTIONS = [
    "Does aspirin reduce the risk of myocardial infarction in patients with cardiovascular disease?",
    "What is the mechanism by which metformin lowers blood glucose in type 2 diabetes?",
    "How do statins reduce LDL cholesterol and lower cardiovascular risk?",
    "What is the role of ACE inhibitors in treating hypertension and heart failure?",
    "How does warfarin prevent blood clots and what are its main risks?",
]

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biomedical Semantic Leakage Detector",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.step-card   { border-radius:6px; padding:8px 12px; margin:3px 0; font-size:0.92em; }
.contra      { background:#fde8e8; border-left:4px solid #e53e3e; }
.entail      { background:#e8f5e9; border-left:4px solid #2e7d32; }
.neutral-box { background:#f5f5f5; border-left:4px solid #9e9e9e; }
.metric-big  { font-size:2em; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─── Parsing helpers ──────────────────────────────────────────────────────────
_LEAD_RE = re.compile(r"^\s*(?:\d+[\.\)]\s*|[-•*]\s*)")

def _parse_steps(text: str) -> List[str]:
    steps = []
    for ln in (text or "").splitlines():
        ln = _LEAD_RE.sub("", ln.strip()).strip()
        if ln:
            steps.append(ln)
    if len(steps) <= 1:
        parts = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", text or "") if p.strip()]
        if len(parts) > 1:
            return parts
    return steps or ["(no steps parsed)"]

# ─── CoT generation via OpenRouter ───────────────────────────────────────────
def _call_openrouter(question: str, model_id: str, api_key: str) -> Tuple[List[str], str, str]:
    """Returns (steps, raw_text, error_msg)."""
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
        resp = client.chat.completions.create(
            model=model_id,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a biomedical reasoning assistant. "
                        "Return ONLY concise numbered reasoning steps (1. 2. 3. ...). "
                        "Be precise and medically accurate. No preamble or closing remarks."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Break down the reasoning into short numbered steps:\n{question}",
                },
            ],
            extra_headers={
                "HTTP-Referer": "https://github.com/biomedical-semantic-leakage",
                "X-Title": "Biomedical Semantic Leakage Detection",
            },
        )
        raw = (resp.choices[0].message.content or "") if resp.choices else ""
        return _parse_steps(raw), raw, ""
    except Exception as exc:
        return [f"[ERROR] {exc}"], "", str(exc)

# ─── Entailment ───────────────────────────────────────────────────────────────
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "it",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "with",
    "by", "from", "that", "this", "which", "these", "those", "its",
}
_POS = {
    "increase", "increases", "promote", "promotes", "activate", "activates",
    "enhance", "enhances", "stimulate", "stimulates", "cause", "causes",
    "upregulate", "upregulates", "potentiate", "potentiates",
}
_ANT = {
    "reduce", "reduces", "lower", "lowers", "inhibit", "inhibits",
    "prevent", "prevents", "block", "blocks", "decrease", "decreases",
    "suppress", "suppresses", "downregulate", "downregulates",
}
_NEG = {"not", "no", "cannot", "fail", "fails", "lack", "lacks", "absent", "never", "without"}


def _heuristic_pair(premise: str, hypothesis: str) -> Dict[str, float]:
    def _tok(s: str):
        return set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) - _STOP

    p_t = _tok(premise)
    h_t = _tok(hypothesis)
    overlap = len(p_t & h_t) / max(len(h_t), 1) if h_t else 0.0
    p_l, h_l = premise.lower(), hypothesis.lower()

    neg_p = any(w in p_l.split() for w in _NEG)
    neg_h = any(w in h_l.split() for w in _NEG)
    pos_p = any(w in p_l for w in _POS)
    ant_h = any(w in h_l for w in _ANT)
    pos_h = any(w in h_l for w in _POS)
    ant_p = any(w in p_l for w in _ANT)

    contra = 0.05
    if neg_p ^ neg_h:
        contra += 0.35
    if (pos_p and ant_h) or (ant_p and pos_h):
        contra += 0.25
    entail = min(0.65, overlap * 0.85)
    neutral = max(0.05, 1.0 - entail - contra)
    total = entail + neutral + contra
    return {
        "entailment":    round(entail  / total, 3),
        "neutral":       round(neutral / total, 3),
        "contradiction": round(contra  / total, 3),
    }


def _run_entailment(steps: List[str], use_model: bool) -> List[Dict[str, Any]]:
    """Score adjacent step pairs. Returns list of pair dicts."""
    if len(steps) < 2:
        return []

    if use_model:
        try:
            from utils.hybrid_checker import build_entailment_records  # type: ignore
            recs = build_entailment_records(steps, [[] for _ in steps]) or []
            out = []
            for r in recs:
                probs = r.get("probs") or {}
                label = str(
                    r.get("final_label") or r.get("label")
                    or (max(probs, key=probs.get) if probs else "neutral")
                )
                sp = r.get("step_pair") or []
                i = int(sp[0]) if sp else int(r.get("i", 0))
                j = int(sp[1]) if sp else int(r.get("j", 1))
                out.append({
                    "i": i, "j": j, "label": label,
                    "E": float(probs.get("entailment", 0.0)),
                    "N": float(probs.get("neutral", 1.0)),
                    "C": float(probs.get("contradiction", 0.0)),
                })
            return out
        except Exception:
            pass  # fall through to heuristic

    pairs = []
    for i in range(len(steps) - 1):
        scores = _heuristic_pair(steps[i], steps[i + 1])
        label = max(scores, key=scores.__getitem__)
        pairs.append({
            "i": i, "j": i + 1, "label": label,
            "E": scores["entailment"],
            "N": scores["neutral"],
            "C": scores["contradiction"],
        })
    return pairs


def _label_css(label: str) -> str:
    return {"contradiction": "contra", "entailment": "entail", "neutral": "neutral-box"}.get(label, "neutral-box")


def _label_color(label: str) -> str:
    return {"contradiction": "#e53e3e", "entailment": "#2e7d32", "neutral": "#9e9e9e"}.get(label, "#9e9e9e")


# ─── Full pipeline ────────────────────────────────────────────────────────────
def _run_pipeline(
    question: str,
    model_name: str,
    model_id: str,
    api_key: str,
    use_model_nli: bool,
) -> Dict[str, Any]:
    t0 = time.time()
    steps, raw, error = _call_openrouter(question, model_id, api_key)
    pairs = _run_entailment(steps, use_model_nli)
    contradictions = sum(1 for p in pairs if p["label"] == "contradiction")
    total_pairs = len(pairs)
    rate = contradictions / total_pairs if total_pairs else 0.0
    max_pc = max((p["C"] for p in pairs), default=0.0)
    return {
        "model_name":        model_name,
        "model_id":          model_id,
        "steps":             steps,
        "raw":               raw,
        "error":             error,
        "pairs":             pairs,
        "contradictions":    contradictions,
        "total_pairs":       total_pairs,
        "contradiction_rate": rate,
        "max_pc":            max_pc,
        "duration_s":        round(time.time() - t0, 2),
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Semantic Leakage")
    st.markdown("*Biomedical CoT Analyzer*")
    st.divider()

    st.markdown("### 🔑 API Keys")
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Get a free key at openrouter.ai/keys — gives access to Claude, GPT-4o, Gemini, Llama.",
    )
    umls_key = st.text_input(
        "UMLS API Key (optional)",
        type="password",
        placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        help="Free at uts.nlm.nih.gov — enables UMLS concept validity scoring.",
    )
    if umls_key:
        os.environ["UMLS_API_KEY"] = umls_key

    st.divider()
    st.markdown("### ⚙️ Settings")
    use_model_nli = st.toggle(
        "Full NLI Model",
        value=False,
        help=(
            "OFF = heuristic NLI (fast, no download). "
            "ON = PubMedBERT-BioNLI-LoRA (accurate, requires model download on first run)."
        ),
    )
    if not use_model_nli:
        os.environ["FORCE_HEURISTIC_NLI"] = "1"
    elif "FORCE_HEURISTIC_NLI" in os.environ:
        del os.environ["FORCE_HEURISTIC_NLI"]

    st.divider()
    st.markdown("### 🤖 Models")
    selected_models: Dict[str, str] = {}
    for name, mid in OPENROUTER_CATALOGUE.items():
        if st.checkbox(name, value=(name in DEFAULT_SELECTED), key=f"model__{name}"):
            selected_models[name] = mid

    st.divider()
    # Status
    key_ok = bool(openrouter_key)
    model_ok = bool(selected_models)
    if key_ok and model_ok:
        st.success(f"✅ Ready — {len(selected_models)} model(s)")
    elif not key_ok:
        st.warning("🔑 Enter your OpenRouter key above")
    else:
        st.warning("☑️ Select at least one model")

    st.divider()
    st.caption(
        "Detects **semantic leakage** in LLM CoT reasoning — where consecutive steps "
        "contradict or drift from earlier claims — using NLI + UMLS ontology."
    )

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🧬 Biomedical Semantic Leakage Detector")
st.caption(
    "Detect contradictions in LLM chain-of-thought reasoning using "
    "Natural Language Inference (NLI) + UMLS biomedical ontology."
)

tab_single, tab_compare, tab_about = st.tabs(
    ["📄 Single Question", "📊 Multi-Model Compare", "ℹ️ About"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single question
# ════════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Analyze a Single Question")

    sample_idx = st.selectbox(
        "Or pick a sample question",
        options=["(type your own below)"] + SAMPLE_QUESTIONS,
        key="sample_q",
    )
    prefill = "" if sample_idx == "(type your own below)" else sample_idx

    question = st.text_area(
        "Biomedical question",
        value=prefill,
        placeholder="e.g. Does aspirin reduce the risk of myocardial infarction?",
        height=90,
        key="single_q_input",
    )

    col_run, col_clear, _ = st.columns([1, 1, 8])
    with col_run:
        run_single = st.button(
            "▶ Run",
            type="primary",
            disabled=not (key_ok and model_ok and question.strip()),
            key="run_single",
        )
    with col_clear:
        if st.button("🗑 Clear", key="clear_single"):
            st.session_state.pop("single_results", None)
            st.rerun()

    if run_single and question.strip():
        results: Dict[str, Any] = {}
        prog = st.progress(0, text="Starting…")
        total = len(selected_models)
        for idx, (mname, mid) in enumerate(selected_models.items()):
            prog.progress(idx / total, text=f"Running {mname}…")
            results[mname] = _run_pipeline(
                question.strip(), mname, mid, openrouter_key, use_model_nli
            )
        prog.progress(1.0, text="Done!")
        time.sleep(0.3)
        prog.empty()
        st.session_state["single_results"] = {
            "question": question.strip(),
            "results": results,
        }

    if "single_results" in st.session_state:
        data = st.session_state["single_results"]
        st.markdown(f"**Question:** *{data['question']}*")
        st.divider()

        for mname, res in data["results"].items():
            rate_pct = res["contradiction_rate"] * 100
            status_icon = "🔴" if rate_pct > 20 else "🟡" if rate_pct > 5 else "🟢"
            header = (
                f"{status_icon} **{mname}** — "
                f"{res['contradictions']}/{res['total_pairs']} contradictions "
                f"({rate_pct:.1f}%)  ·  {len(res['steps'])} steps  ·  {res['duration_s']}s"
            )
            with st.expander(header, expanded=True):
                if res.get("error"):
                    st.error(f"API error: {res['error']}")
                    continue

                col_steps, col_pairs = st.columns([3, 2])

                with col_steps:
                    st.markdown("**Reasoning Steps**")
                    for i, step in enumerate(res["steps"]):
                        pair = next((p for p in res["pairs"] if p["i"] == i), None)
                        css = _label_css(pair["label"]) if pair else "neutral-box"
                        st.markdown(
                            f'<div class="step-card {css}">'
                            f'<strong>{i + 1}.</strong> {step}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                with col_pairs:
                    st.markdown("**Pairwise NLI (step i → i+1)**")
                    if res["pairs"]:
                        import pandas as pd

                        df = pd.DataFrame(
                            [
                                {
                                    "Pair":  f"{p['i'] + 1}→{p['j'] + 1}",
                                    "Label": p["label"][0].upper(),
                                    "E":     f"{p['E']:.2f}",
                                    "N":     f"{p['N']:.2f}",
                                    "C":     f"{p['C']:.2f}",
                                }
                                for p in res["pairs"]
                            ]
                        )

                        def _style_label(val: str) -> str:
                            return (
                                "color:red;font-weight:bold" if val == "C"
                                else "color:green;font-weight:bold" if val == "E"
                                else "color:grey"
                            )

                        st.dataframe(
                            df.style.map(_style_label, subset=["Label"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption("No pairs — need ≥ 2 steps.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-model compare
# ════════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Multi-Model Comparison")
    st.caption("Run the same question across all selected models and compare contradiction rates side by side.")

    sample_idx2 = st.selectbox(
        "Pick a sample question",
        options=["(type your own below)"] + SAMPLE_QUESTIONS,
        key="sample_q2",
    )
    prefill2 = "" if sample_idx2 == "(type your own below)" else sample_idx2

    compare_q = st.text_area(
        "Biomedical question",
        value=prefill2,
        placeholder="e.g. What is the mechanism by which metformin lowers blood glucose in type 2 diabetes?",
        height=90,
        key="compare_q_input",
    )

    col_run2, col_clear2, _ = st.columns([1.5, 1, 7.5])
    with col_run2:
        run_compare = st.button(
            "▶ Compare All Models",
            type="primary",
            disabled=not (key_ok and model_ok and compare_q.strip()),
            key="run_compare",
        )
    with col_clear2:
        if st.button("🗑 Clear", key="clear_compare"):
            st.session_state.pop("compare_results", None)
            st.rerun()

    if run_compare and compare_q.strip():
        comp_results: Dict[str, Any] = {}
        prog2 = st.progress(0, text="Starting…")
        total = len(selected_models)
        for idx, (mname, mid) in enumerate(selected_models.items()):
            prog2.progress(idx / total, text=f"Running {mname}…")
            comp_results[mname] = _run_pipeline(
                compare_q.strip(), mname, mid, openrouter_key, use_model_nli
            )
        prog2.progress(1.0, text="Done!")
        time.sleep(0.3)
        prog2.empty()
        st.session_state["compare_results"] = {
            "question": compare_q.strip(),
            "results": comp_results,
        }

    if "compare_results" in st.session_state:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        data = st.session_state["compare_results"]
        results = data["results"]
        st.markdown(f"**Question:** *{data['question']}*")
        st.divider()

        model_names  = list(results.keys())
        contra_rates = [results[m]["contradiction_rate"] * 100 for m in model_names]
        n_steps      = [len(results[m]["steps"]) for m in model_names]
        max_pcs      = [results[m]["max_pc"] * 100 for m in model_names]

        # ── Summary charts ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        bar_colors = [
            "#e53e3e" if r > 20 else "#f6a623" if r > 5 else "#2e7d32"
            for r in contra_rates
        ]
        axes[0].bar(model_names, contra_rates, color=bar_colors, edgecolor="white")
        axes[0].set_title("Contradiction Rate (%)", fontweight="bold")
        axes[0].set_ylabel("Contradiction Rate (%)")
        axes[0].set_ylim(0, max(max(contra_rates) * 1.25, 5))
        for i, v in enumerate(contra_rates):
            axes[0].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=15)

        axes[1].bar(model_names, n_steps, color="#4299e1", edgecolor="white")
        axes[1].set_title("Reasoning Chain Length (steps)", fontweight="bold")
        axes[1].set_ylabel("Steps")
        for i, v in enumerate(n_steps):
            axes[1].text(i, v + 0.1, str(v), ha="center", fontsize=9, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=15)

        axes[2].bar(model_names, max_pcs, color="#9b59b6", edgecolor="white")
        axes[2].set_title("Max P(Contradiction) (%)", fontweight="bold")
        axes[2].set_ylabel("Max P(C) (%)")
        axes[2].set_ylim(0, max(max(max_pcs) * 1.25, 5))
        for i, v in enumerate(max_pcs):
            axes[2].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        axes[2].tick_params(axis="x", rotation=15)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # ── P(contradiction) heatmap ──────────────────────────────────────────
        st.markdown("**P(contradiction) Heatmap — step pairs × models**")
        max_pairs = max((len(results[m]["pairs"]) for m in model_names), default=0)
        if max_pairs > 0:
            matrix = np.zeros((max_pairs, len(model_names)))
            cell_labels = [[""] * len(model_names) for _ in range(max_pairs)]
            for col_i, mname in enumerate(model_names):
                for row_i, pair in enumerate(results[mname]["pairs"]):
                    matrix[row_i, col_i] = pair["C"]
                    lbl = "C" if pair["label"] == "contradiction" else "N"
                    cell_labels[row_i][col_i] = f"{lbl}\n{pair['C']:.2f}"

            fig2, ax = plt.subplots(
                figsize=(max(8, len(model_names) * 2.2), max(4, max_pairs * 0.55))
            )
            im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
            ax.set_yticks(range(max_pairs))
            ax.set_yticklabels([f"{i + 1}→{i + 2}" for i in range(max_pairs)], fontsize=8)
            for r in range(max_pairs):
                for c in range(len(model_names)):
                    txt = cell_labels[r][c]
                    if txt:
                        ax.text(
                            c, r, txt,
                            ha="center", va="center", fontsize=7,
                            color="white" if matrix[r, c] > 0.55 else "black",
                        )
            plt.colorbar(im, ax=ax, label="P(contradiction)")
            ax.set_title("P(contradiction) per step pair per model", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.divider()

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("**Summary Table**")
        summary_rows = []
        for mname in model_names:
            res = results[mname]
            summary_rows.append({
                "Model":         mname,
                "Steps":         len(res["steps"]),
                "Pairs":         res["total_pairs"],
                "Contradictions": res["contradictions"],
                "Rate":          f"{res['contradiction_rate'] * 100:.1f}%",
                "Max P(C)":      f"{res['max_pc']:.2f}",
                "Time (s)":      res["duration_s"],
            })
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ── Download ──────────────────────────────────────────────────────────
        export = {
            "question": data["question"],
            "results": {
                k: {
                    "model_id":          v["model_id"],
                    "steps":             v["steps"],
                    "pairs":             v["pairs"],
                    "contradictions":    v["contradictions"],
                    "total_pairs":       v["total_pairs"],
                    "contradiction_rate": v["contradiction_rate"],
                    "max_pc":            v["max_pc"],
                    "duration_s":        v["duration_s"],
                    "error":             v.get("error", ""),
                }
                for k, v in results.items()
            },
        }
        st.download_button(
            "⬇ Download Results (JSON)",
            data=json.dumps(export, indent=2),
            file_name="leakage_results.json",
            mime="application/json",
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ════════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("About This Tool")
    st.markdown("""
**Biomedical Semantic Leakage Detector** identifies logical inconsistencies in LLM
chain-of-thought (CoT) reasoning for biomedical questions.

---

### What is Semantic Leakage?

Semantic leakage occurs when consecutive reasoning steps in an LLM's CoT output contradict
each other — for example, first asserting *"aspirin reduces platelet aggregation and lowers MI
risk"* and then asserting *"aspirin does not reliably reduce MI risk in secondary prevention"*
without acknowledging the shift. In clinical AI, such contradictions undermine trust in model
reasoning.

---

### Pipeline

| Stage | Description |
|-------|-------------|
| **1. CoT Generation** | Question → LLM (via OpenRouter) → numbered reasoning steps |
| **2. Concept Extraction** | Surface n-grams from each step → UMLS CUI linking (if key provided) |
| **3. Entailment Checking** | Adjacent step pairs scored with heuristic NLI or PubMedBERT-BioNLI-LoRA |
| **4. UMLS Adjustment** | CUI Jaccard overlap between step pairs adjusts E/N/C probabilities |
| **5. Report** | Contradiction rate, P(C) heatmap, guard signals per model |

---

### NLI Labels

| Label | Meaning |
|-------|---------|
| **E — Entailment** | Step i+1 logically follows from step i |
| **N — Neutral** | Steps are related but neither entail nor contradict |
| **C — Contradiction** | Step i+1 conflicts with or reverses step i |

---

### NLI Modes

- **Heuristic (default, fast):** Token overlap → entailment; negation / direction-verb
  mismatch → contradiction. No model download required.
- **Full Model (accurate):** `Bam3752/PubMedBERT-BioNLI-LoRA` — fine-tuned LoRA biomedical
  NLI. Downloads on first use (~400 MB). Toggle "Full NLI Model" in the sidebar.

---

### Getting API Keys

- **OpenRouter**: Free at [openrouter.ai/keys](https://openrouter.ai/keys) — single key
  for Claude, GPT-4o, Gemini, Llama and 200+ models.
- **UMLS**: Free at [uts.nlm.nih.gov](https://uts.nlm.nih.gov) — enables concept validity
  scoring and CUI Jaccard overlap for ontology-aware entailment.

---

### References

- *Ontology-Guided Semantic Leakage Detection in Biomedical Chain-of-Thought Reasoning*
- NLI model: `Bam3752/PubMedBERT-BioNLI-LoRA`
- UMLS: Unified Medical Language System (National Library of Medicine)
""")
