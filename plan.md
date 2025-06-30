# Synthetic Trace Generation Plan for Tool‑Augmented Summariser

## Overview

We will build a corpus of tool‑augmented training traces that demonstrate strategic use of two custom tools during summary generation:

- **/** – regex/BM25 retrieval over the source document.
- &#x20;– on‑the‑fly deletion (sentence or clause) followed by a corrected rewrite.

Traces show the model *why* and *how* to call the tools, enabling ReTool‑style RL‑VR fine‑tuning with FENICE + rule gates.

---

## Four‑Step Data Pipeline

1. **Draft faulty summaries**
   - Generate initial summaries with Llama‑3‑1B (hallucination‑prone).
   - Seed with adversarial prompts to boost error diversity.
2. **GPT‑4 fact‑check & annotation**
   - GPT‑4 detects each factual error, returns structured JSON:
     ```json
     {"error_span":"…","corrected_clause":"…","regex_key":"…"}
     ```
   - Post‑processor injects:
     ```
     <search>/(regex_key)/</search>
     <results><0>…</0></results>
     <delete scope="sentence"/>
     corrected_clause
     ```
3. **Continue generation**
   - Feed edited context back to the 1 B model to complete the remaining summary.
4. **Positive “search‑only” traces**
   - For \~25 % of *correct* sentences, insert `<search>`+`<results>` **without** deletion to teach restraint.

---

## Quality‑Control Filters

| Filter               | Rule                                             |
| -------------------- | ------------------------------------------------ |
| **Result relevance** | Drop if top‑1 BM25 < 0.3 cosine                  |
| **FENICE gain**      | Keep trace only if FENICE ↑ ≥ 0.05 after rewrite |
| **Token budget**     | Truncate `<results>` to ≤ 200 tokens             |
| **Diversity**        | MinHash dedup; Jaccard ≥ 0.9                     |

Target noise ≤ 5 % (spot‑check 100/5 k).

---

## Cost & Latency Estimate (per 10 k traces)

- Drafting (1 B LL‑3): **\~0.3 h GPU**
- GPT‑4 auditing: **\~\$300, 3–6 s/doc** (cut 35–45 % via rule pre‑filter)
- Total for 50 k traces: **≈ \$1.5 k–2 k + 2 GPU‑days**

---

## Risk Mitigations

- **Sparse early rewards** → curriculum (looser thresholds first 1 k PPO iters).
- **Delete loops** → hard cap of 5 deletes/doc + penalty.
- **Cache resets after deletion** → re‑tokenise remaining context à la ReTool.
- **Tool spam** → budget penalty: −0.05 × (#search + #delete).

---

## Phased Implementation Roadmap

### Phase A – Baseline RL‑VR (no tools)

1. **Implement** a vanilla PPO/GRPO loop that uses only the hard rule‑bundle reward (length, entity overlap, number consistency, no profanity, etc.).
2. **Train** on a small corpus (e.g., 50 k CNN‑DailyMail articles) until the model achieves ≥ 20 % rule‑pass rate.
3. **Evaluate** with ROUGE + rule metrics; establish dashboards and checkpoints.

### Phase B – Integrate FENICE Reward

1. **Plug** the distilled FENICE scorer into the reward: `R = 0.7 × FENICE + 0.3 × Rules`.
2. **Curriculum‑schedule** the FENICE threshold (τ): start at 0.50, tighten to 0.65 once pass‑rate > 20 %.
3. **Re‑train**; track hallucination reduction and stability.

### Phase C – Synthetic Traces + SFT for Tool Syntax

1. **Generate** a 5 k pilot batch using the four‑step data pipeline, refine filters; then scale to 50 k traces.
2. **Supervised‑fine‑tune** the base model for 2‑3 epochs on these traces so it learns `<search>`/`<delete>` syntax and basic tool timing.
3. **Validate**: at least 95 % of sampled generations must produce syntactically correct tool calls.

### Phase D – Tool‑Aware RL‑VR (ReTool style)

1. **Enable execution sandbox** so `<search>` retrieves snippets and `<delete>` prunes context during roll‑outs.
2. **Add** tool‑budget term: `–0.05 × (#search + #delete)`.
3. **Fine‑tune with PPO** using the combined FENICE + rule reward.
4. **Ablate**: run variants without delete, without search, and without budget penalty to measure contribution.

### Phase E – Distillation & Deployment

1. **Distil** the final 7 B/13 B policy into a 3 B “fast” model via regression on thought‑hidden traces.
2. **Benchmark** latency (< 100 ms/summary) and factuality parity on hidden test sets.
3. **Package & deploy** inference endpoint; set up continuous evaluation with the rule/FENICE checker.

## Library & Tooling Stack

| Purpose                          | Preferred Lightweight Tool / Library                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| **Core DL**                      | • PyTorch ≥ 2.3  • HuggingFace Transformers ≥ 4.42                                           |
| **RL Framework**                 | • HuggingFace TRL (PPO/GRPO)  • *ReTool* engine for tool‑aware PPO (uses TRL under the hood) |
| **Experiment Tracking**          | • Weights & Biases (wandb) – system‑wide metrics, model artefacts                            |
| **Fast Inference / Sampling**    | • vLLM for baseline & PPO roll‑outs (supports paged KV, Flash‑Attention‑2)                   |
| **Quantisation / Memory**        | • bitsandbytes 8‑bit/4‑bit  • accelerate for multi‑GPU                                       |
| **Tokenizer & NLP utils**        | • spaCy 3 (sentence split, NER)  • regex (std lib)  • rapidfuzz for fuzzy match              |
| **Retrieval (for ****\`\`****)** | • rank‑bm25 on pre‑tokenised sentences  • Fallback simple regex scan                         |
| **Sandbox Execution**            | • Lightweight Python subprocess pool (ReTool style) with 0.3 s timeout                       |
| **FENICE Components**            | • DeBERTa‑v3‑large via HF hub for NLI  • Distilled T5‑base for claim extraction              |
| **Data & Config**                | • datasets + pandas  • Hydra for config mgmt                                                 |

> *All tools have stable, pure‑Python wheels; no heavyweight search servers (e.g., Elasticsearch) unless latency tests demand it.*

---

### Milestones & Go/No‑Go Criteria

| ID | Milestone          | Success Threshold                               |
| -- | ------------------ | ----------------------------------------------- |
| M0 | Rule‑only baseline | ≥ 20 % rule passl                               |
| M1 | +FENICE            | Hallucinations ↓ ≥ 40 % vs. M0                  |
| M2 | Tool‑aware RL      | Combined pass ≥ 60 %                            |
| M3 | Distilled model    | Latency ≤ 100 ms & factuality within 2 pp of M2 |

Risk mitigations, quality‑control filters and cost estimates remain as outlined above.

