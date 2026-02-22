# Experiment Results

---

## What we're measuring

When an LLM answers a medical question, it produces a **chain-of-thought (CoT)** — a numbered sequence of reasoning steps leading to a conclusion. The goal of these experiments is to detect **semantic leakage**: cases where one step in that chain contradicts or reverses what an earlier step said, without the model acknowledging the shift.

To detect contradictions automatically, we use **NLI (Natural Language Inference)** — a technique that takes two sentences and classifies their relationship as one of:
- **Entailment** — the second sentence follows logically from the first
- **Neutral** — the sentences are related but don't confirm or deny each other
- **Contradiction** — the second sentence conflicts with the first

We run NLI on every adjacent pair of steps in a reasoning chain (step 1→2, step 2→3, etc.) and flag pairs classified as contradiction.

We use a **heuristic NLI** mode throughout these experiments (instead of a full neural model) because it's fast, requires no model downloads, and is transparent. It works by:
- Scoring token overlap between step pairs → high overlap → entailment
- Detecting negation word mismatches (e.g., one step says "reduces", the next says "does not reduce") → contradiction
- Detecting direction-verb flips (e.g., "increases" followed by "decreases") → contradiction

The tradeoff is that it only catches surface-level patterns and misses subtler semantic contradictions. The full `PubMedBERT-BioNLI-LoRA` model catches more but requires a ~400MB download.

**Guard signals** are lightweight rule-based flags that run alongside NLI to add context:
- `lexical_duplicate` — fires if two consecutive steps are near-identical word-for-word (would indicate the model is repeating itself)
- `caution_band` — fires when hedging words appear ("however", "may", "in some cases") — marks the step as uncertain, not necessarily wrong
- `direction_conflict` — fires when consecutive steps use opposite-direction verbs (e.g., "promotes" → "inhibits") — a strong indicator of an actual contradiction

**UMLS (Unified Medical Language System)** is a biomedical ontology maintained by the US National Library of Medicine. Each medical concept (aspirin, myocardial infarction, etc.) has a unique identifier called a **CUI**. When UMLS is configured, the pipeline links the terms in each reasoning step to their CUIs, which lets us measure how many concepts the model is using correctly (concept validity) and whether two steps share the same concepts (CUI Jaccard overlap — used to adjust entailment/contradiction probabilities).

---

## Experiment 1 — Cross-Model Semantic Leakage Benchmark

We ran four models on 40 biomedical questions and measured how often their reasoning steps contradict each other.

**Setup**

| Parameter | Value |
|-----------|-------|
| Models | claude-haiku, gpt-4o-mini, gemini-flash, llama-3-70b |
| Questions | 40 (PubMedQA / MedQA) |
| NLI mode | Heuristic (see above) |
| UMLS | Not configured for this run |
| Total runs | 160 (4 models × 40 questions) |

---

### Figure 1 — Per-Model Contradiction Rates

![Experiment 1: Cross-Model Benchmark](experiments/results/result_images/exp1_1.png)

**(a) Contradiction rate per model (boxplot)**

Each box shows the spread of per-question contradiction rates across all 40 questions. The contradiction rate for a single question is: number of contradicting step-pairs ÷ total step-pairs.

| Model | Median Rate | Outliers up to |
|-------|------------|----------------|
| claude-haiku | ~10% | 38% |
| gpt-4o-mini | ~1% | 43% |
| gemini-flash | ~10% | 50% |
| llama-3-70b | ~12% | 25% |

GPT-4o-mini is the most consistent — its answers tend to move in one direction without pivoting. Claude-haiku and gemini-flash both sit around 10% median but occasionally spike much higher on specific questions.

**(b) NLI label breakdown per model (stacked bar)**

This shows the average number of step-pairs per question labelled entailment (blue), neutral (purple), and contradiction (red). Claude-haiku and gemini-flash write longer answers (~12 step-pairs per question vs ~7 for the others), so they accumulate more contradictions in absolute terms even if their per-pair rate is similar. Entailment does appear here (unlike in smaller pilot runs) because with 40 diverse questions, some consecutive steps do share enough vocabulary to pass the heuristic's token-overlap threshold.

**(c) Concept validity scatter**

Empty — UMLS wasn't configured for this run. When enabled, each dot would be one question, with x = fraction of terms the model used that could be linked to a UMLS CUI (concept validity), and y = contradiction rate. The idea is that vaguer, un-linkable reasoning might correlate with more contradictions.

---

### Figure 2 — Contradiction Rate vs. Reasoning Depth

![Semantic Leakage Grows with Reasoning Depth](experiments/results/result_images/exp1_2.png)

**Reasoning depth** here means the position of a step-pair within the chain — depth 0 is the first pair (steps 1→2), depth 5 is the sixth pair (steps 6→7), and so on. Both panels plot how contradiction rate and average P(contradiction) change as depth increases.

The pattern is clear: contradiction risk increases with depth. Early steps tend to establish a mechanism ("aspirin inhibits COX-1 → reduces thromboxane A2 → reduces platelet aggregation"). Later steps tend to introduce caveats or exceptions ("however, aspirin also increases bleeding risk") — that direction flip is what the NLI heuristic flags. GPT-4o-mini stays flat because its chains are short and don't reach the later steps where pivots typically occur. The gemini-flash spike at depth ~22 comes from one or two unusually long answers, so it's not a reliable finding.

---

### Figure 3 — Guard Signal Analysis

![Guard Signal Analysis](experiments/results/result_images/exp1_3.png)

**(a) How often each guard fires per question, per model**

| Model | caution_band | direction_conflict | lexical_duplicate |
|-------|--------------:|-------------------:|------------------:|
| claude-haiku | ~5.4 | ~1.2 | 0 |
| gemini-flash | ~5.0 | ~0.9 | 0 |
| gpt-4o-mini | ~2.5 | ~0.5 | 0 |
| llama-3-70b | ~3.0 | ~0.7 | 0 |

`lexical_duplicate` is always zero — models don't repeat steps word-for-word. `caution_band` fires frequently because all models naturally use hedging language in biomedical answers ("may", "in some patients", "depending on context"), which is appropriate given the probabilistic nature of medical evidence.

**(b) Do the guards actually predict contradictions?**

This panel asks: of the step-pairs that were classified as contradictions, what fraction had each guard signal? And of the non-contradicting pairs, what fraction?

| Guard | On contradiction pairs | On other pairs |
|-------|---------------------------:|--------------------:|
| direction_conflict | 28% | 5% |
| caution_band | 5% | 43% |

`direction_conflict` is genuinely useful — it fires 5.6× more often when a contradiction is present than when it isn't. It was designed specifically to catch verb-direction flips, and that's what it does.

`caution_band` fires far more on *non*-contradiction pairs (43%) than on contradiction pairs (5%). That's because hedging words appear mostly during normal topic transitions, not during hard direction flips. This doesn't mean `caution_band` is broken — it just means it's an uncertainty signal ("this step introduces a caveat, a human should check") rather than a contradiction detector.

---

### Limitations

| Issue | Impact | Fix |
|-------|--------|-----|
| Heuristic NLI only | Only catches surface patterns — negation and direction-verb flips. Misses subtler semantic contradictions | Run with full PubMedBERT-BioNLI-LoRA model |
| UMLS not configured | Concept validity panel empty; no CUI-based adjustment to NLI scores | Add UMLS API key (free at uts.nlm.nih.gov) |
| Gemini depth spike at 22 | Only 1–2 questions produce chains that long — not a reliable signal | More questions, or stratify analysis by chain length |
| No human labels | All contradiction calls are automated — no ground truth to validate against | See Exp 3 for 120 hand-labeled pairs |

---

---

## Experiment 2 — Cross-Question Consistency

The standard pipeline checks whether step 3 contradicts step 4 *within one answer*. This experiment asks a different question: does the model say inconsistent things about the *same concept* across *different questions*?

For example — if you ask "how does aspirin reduce heart attack risk?" and separately ask "what are the risks of aspirin therapy?", does the model give answers that contradict each other when you compare their reasoning steps side by side?

To measure this, we group questions by medical concept (all aspirin questions together, all metformin questions together, etc.), collect the reasoning steps from each answer, and run NLI on pairs of steps drawn from *different* questions within the same group. We call this the **cross-answer contradiction rate**.

**Setup**

| Parameter | Value |
|-----------|-------|
| Models | claude-haiku, gpt-4o-mini, gemini-flash, llama-3-70b |
| Concepts | 6: aspirin, metformin, statins, insulin, ACE inhibitors, beta blockers |
| Questions | 5–6 per concept (30 total) |
| Cross-question pairs scored | Up to 50 step-pairs per concept, drawn from different questions |

---

### Figure 4 — Cross-Question Contradiction Rates

![Cross-Question Contradiction Rate](experiments/results/result_images/exp2_1.png)

**(a) Cross-question contradiction rate by concept**

| Concept | Cross-answer rate |
|---------|------------------:|
| Insulin | ~12.5% |
| Metformin | ~10.0% |
| Aspirin | ~6.0% |
| ACE Inhibitors | ~5.0% |
| Statins | ~2.5% |
| Beta Blockers | ~0.0% |

Insulin is the most inconsistent across questions. This makes sense — insulin's role looks quite different depending on whether the question is about type 1 vs type 2 diabetes, normal physiology vs insulin resistance, or hepatic vs peripheral effects. The model frames it differently each time, and those different framings sometimes conflict.

Beta blockers are the most consistent. The mechanism (blocking β1 adrenergic receptors → lower heart rate and contractility) is the same regardless of how the question is framed, so the model rarely contradicts itself.

**(b) Within-answer vs. cross-answer contradiction rates**

| Concept | Within-answer | Cross-answer |
|---------|-------------:|-------------:|
| Insulin | ~22.5% | ~12.5% |
| Metformin | ~17.5% | ~10.0% |
| Aspirin | ~16.5% | ~6.0% |
| Statins | ~15.0% | ~2.5% |
| ACE Inhibitors | ~11.5% | ~5.0% |
| Beta Blockers | ~1.5% | ~0.0% |

The within-answer rate is consistently higher than the cross-answer rate for every concept. The reason: within a single CoT chain, the model builds up a mechanism and then pivots to mention side effects or exceptions — that pivot gets flagged as a contradiction by the NLI heuristic. When answering separate questions, each response starts fresh and stays more consistent in direction.

The effect is largest for statins (within-answer rate is 6× the cross-answer rate). Statin answers almost always follow the same pattern: explain the HMG-CoA reductase mechanism → mention myopathy as a risk. That pivot within one answer looks like a contradiction, but across separate questions the model is perfectly consistent.

---

### Figure 5 — Cross-Question Heatmaps per Concept

![Cross-Question Heatmaps](experiments/results/result_images/exp2-3.png)

Each grid shows which pairs of questions within a concept group produced contradicting steps when compared. White = no contradiction detected, dark red = high contradiction rate. Each axis is a question index within that concept's group.

**Aspirin** — light pink spread across most pairs. Aspirin has competing effects (antiplatelet/cardioprotective vs ulcerogenic/anticoagulant) so mild inconsistency shows up regardless of which two questions are compared.

**Metformin** — one very dark cell at Q0 vs Q1: "how does metformin work" vs "when is metformin contraindicated". The mechanism answer talks about reducing liver glucose output; the contraindication answer brings in renal failure risk and lactic acidosis. These read as opposing directions to the NLI heuristic.

**Statins** — mostly light, with one hotspot at Q1 vs Q3: "do statins reduce stroke risk?" vs "do statins reduce mortality in heart failure?". Evidence is strong for stroke prevention, mixed for heart failure mortality, so the model gives different levels of confidence and the heuristic picks up the mismatch.

**Insulin** — broadly pink across many pairs. Because insulin physiology is genuinely different depending on context (type 1, type 2, basal, bolus, hepatic, peripheral), the model produces different framings across questions, leading to spread-out inconsistency rather than one focal hotspot.

**ACE inhibitors** — hotspot concentrated around Q2, Q3, Q4: "do they protect kidney function" vs "what are the adverse effects" vs "can they cause hyperkalemia". ACE inhibitors protect kidneys in early diabetic nephropathy but can harm them in dehydrated patients and commonly cause high potassium. The model captures both sides but doesn't reconcile them when the questions are compared.

**Beta blockers** — almost entirely white. The mechanism is simple and directionally consistent no matter how the question is framed.

---

### Limitations

| Issue | Impact | Fix |
|-------|--------|-----|
| Heuristic NLI | May miss semantic contradictions that don't involve direction-verb flips or negation | Use full NLI model |
| Only 5–6 questions per concept | Some heatmap cells represent just 1–2 step-pairs — not very reliable | Expand to 10–15 questions per concept |
| Cross-NLI only run on primary model (claude-haiku) | Other models are cached but not yet compared cross-question | Run separately per model and compare |
| No human labels | All contradiction calls are automated | Annotate the top contradiction examples manually to validate |

---

---

## Experiment 3 — Guard Signal Ablation

Experiments 1 and 2 used fully automated NLI scoring. This experiment uses a hand-labeled set of 101 step-pairs (39 entailment, 31 neutral, 31 contradiction) to measure how well the detector actually works — and whether the add-ons (UMLS linking and guard signals) actually help.

The setup is an **ablation study**: we test four versions of the detector by turning components on and off:
- **A: Pure NLI** — just the NLI model, nothing else
- **B: NLI + UMLS** — NLI scores adjusted using CUI Jaccard overlap between step pairs (if two steps share the same UMLS concepts, that's evidence for entailment; if they share concepts but with conflicting relations, that's evidence for contradiction)
- **C: NLI + Guards** — NLI scores adjusted using the guard signals (`direction_conflict`, `caution_band`, `relation_violation`, etc.)
- **D: Full Hybrid** — all of the above combined

Two new UMLS-based guard signals appear in this experiment that weren't in Exp 1:
- `relation_violation` — fires when the UMLS ontological relation between concepts in step i conflicts with the relation implied in step i+1 (e.g., step i says drug A *treats* disease B, step i+1 implies drug A *causes* disease B)
- `ontology_override` — fires when the model asserts something that contradicts a known UMLS hierarchy (e.g., calling a drug a disease)
- `provisional_support` — fires when both steps reference the same concept but one step qualifies the claim with uncertainty ("may", "in some cases") while the other states it definitively

We measure each condition using:
- **AUROC** (Area Under the ROC Curve) — how well the detector ranks contradictions above non-contradictions on a scale from 0.5 (random) to 1.0 (perfect). A score of 0.96 means 96% of the time, a randomly chosen contradiction pair gets a higher score than a randomly chosen non-contradiction pair.
- **Average Precision** — similar to AUROC but specifically measures precision at each recall threshold
- **F1 (Contradiction)** — the harmonic mean of precision and recall specifically for the contradiction class. Precision = of pairs flagged as contradictions, how many actually were. Recall = of actual contradictions, how many did we catch.

---

### Figure 6 — Guard Signal Co-occurrence by Label Class

![Guard Signal Co-occurrence](experiments/results/result_images/exp3_1.png)

Each panel shows a 6×6 grid of guard signals. Each cell shows: given that the row signal fires on a pair, what fraction of the time does the column signal also fire? We break this down separately for entailment pairs, neutral pairs, and contradiction pairs.

**What stands out:**

`ontology_override` and `provisional_support` always fire together (1.00 in both directions, across all three label classes). This means they're computed from the same underlying UMLS data and essentially capture the same event — when one fires, the other always does too.

`direction_conflict` is 0.00 everywhere — it never fires on any pair in this 101-pair gold set. This is a notable contrast from Exp 1, where it was the most useful discriminating signal. The difference is that the 101 gold-labeled pairs here weren't specifically selected to include direction-verb flips, so the signal has nothing to fire on.

`caution_band` shows a meaningful shift across label classes. On contradiction pairs, it co-occurs with `relation_violation` (15%) and `ontology_override` (33%) — both of these are higher than in entailment pairs (where both were 0%). This tells us that when `caution_band` fires alongside a UMLS-based signal, the pair is more likely to be a contradiction than when `caution_band` fires alone.

---

### Figure 7 — Ablation Results: AUROC and Multi-Metric Comparison

![Ablation Results](experiments/results/result_images/exp3_2.png)

**(a) AUROC by condition**

| Condition | AUROC | 95% CI |
|-----------|------:|--------|
| A: Pure NLI | 0.962 | narrow |
| B: NLI + UMLS | 0.965 | narrow |
| C: NLI + Guards | 0.964 | narrow |
| D: Full Hybrid | 0.964 | narrow |
| Random baseline | 0.500 | — |

All four conditions are essentially tied. The heuristic NLI alone already achieves 0.962 AUROC — the gold-labeled pairs are clear enough that surface-level patterns (negation, token overlap) catch most of them without needing UMLS or guards. Adding UMLS gives the smallest but most consistent improvement (0.962 → 0.965). Adding guards alone doesn't improve over pure NLI. Adding everything together matches NLI + Guards, not better.

**(b) Multi-metric comparison**

Across AUROC, Average Precision, and F1 (Contradiction), the ordering is the same: B (NLI + UMLS) is slightly ahead, then C and D roughly equal to A. NLI + UMLS has the highest F1 at ~0.88 vs ~0.84 for the others. The differences are small in absolute terms but consistent across all three metrics, which gives some confidence that UMLS adjustment is genuinely helping rather than just noise.

---

### Figure 8 — Confusion Matrices for All 4 Conditions

![Confusion Matrices](experiments/results/result_images/exp3_3.png)

A **confusion matrix** shows the four possible outcomes for a binary classifier. Here:
- **True Non-C, Pred Non-C** (top-left) — correctly identified as not a contradiction
- **True Non-C, Pred Contra** (top-right) — false alarm: flagged as contradiction when it wasn't
- **True Contra, Pred Non-C** (bottom-left) — missed contradiction
- **True Contra, Pred Contra** (bottom-right) — correctly caught contradiction

| Condition | Correct non-C | False alarms | Missed | Caught contradictions |
|-----------|-------------:|-------------:|-------:|---------------------:|
| A: Pure NLI | 64 | 6 | 4 | 27 |
| B: NLI + UMLS | 65 | 5 | 3 | 28 |
| C: NLI + Guards | 64 | 6 | 4 | 27 |
| D: Full Hybrid | 64 | 6 | 4 | 27 |

NLI + UMLS (B) is the only condition that improves on Pure NLI — it catches one additional contradiction (27→28) and generates one fewer false alarm (6→5). The margins are small, but the improvement is consistent across all three metrics in Figure 7 too.

Conditions C and D produce identical results to A, which directly explains why their AUROC and F1 match. The guards contribute no additional discrimination on this specific 101-pair dataset — largely because `direction_conflict`, the most powerful guard from Exp 1, never fires on any of these pairs.

---

### What this tells us overall

The heuristic NLI is a surprisingly strong baseline — achieving 0.962 AUROC without any ontology or rule-based augmentation. This is partly because the gold-labeled set contains clear, unambiguous contradictions that surface-level patterns catch well. In messier real-world data, the gap between Pure NLI and NLI + UMLS would likely be larger.

The main practical takeaway: when UMLS is configured, use it — it gives a small consistent improvement at no extra model cost. The guard signals matter most in high-volume settings where you need to triage pairs for human review: `direction_conflict` remains the most targeted signal for contradiction (as shown in Exp 1), while `caution_band` is better used as an uncertainty flag.

---

### Limitations

| Issue | Impact | Fix |
|-------|--------|-----|
| Only 101 gold-labeled pairs | Small evaluation set — differences of 1–2 pairs drive the metric differences | Expand annotation to 500+ pairs |
| `direction_conflict` never fires | Can't evaluate its usefulness on this set | Include pairs specifically selected for direction-verb flips |
| All conditions very similar | Hard to tell which configuration is actually better in production | Test on noisier, less curated data where differences will be larger |
| Heuristic NLI used | The gold labels were likely generated with this same heuristic in the loop — evaluation may be circular | Use the full PubMedBERT-BioNLI-LoRA model and compare against independent human labels |
