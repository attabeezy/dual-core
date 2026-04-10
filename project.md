# SOMAX — Project Reference
**Eliminating the Tokenization Tax for African Languages via Dual-Stream Processing**
**Status:** Implementation Phase (April 2026) | **Hardware:** Cloud (Colab T4) → Edge (Dell Latitude 7400)

---

## 1. Vision

SOMAX is a research-to-production framework designed to eliminate the **"Tokenization Tax"** — the structural inefficiency where African languages require significantly more tokens than English, leading to higher latency, increased cost, and degraded reasoning.

By leveraging **Google WAXAL** (spontaneous Twi/Akan ASR, ~100k) and the **Ghana NLP Pristine-Twi dataset** (~999k clean formal Twi text), SOMAX introduces a dual-stream processing architecture that treats spontaneous speech and formal text as fundamentally different linguistic regimes, sharing a unified vocabulary for maximum embedding efficiency.

---

## 2. Problem Statement

Modern LLM tokenizers are optimized for English-heavy corpora. For African languages like Twi (Akan):

- An English sentence (~10 tokens) can become 40+ tokens in the target language.
- **Consequences:** Higher latency (edge devices), higher costs, weaker reasoning (semantic fragmentation).
- **Metric:** Token Fertility — $F = \text{Tokens} / \text{Words}$
- **Goal:** Reduce $F$ by $\ge 30\%$ through dual-stream vocabulary redesign.

---

## 3. Core Insight: Linguistic Duality

Two Twi data sources capture fundamentally different linguistic distributions:

1. **WAXAL `aka_asr` (Spontaneous):** Noisy, code-switching (e.g., Twi + English), fillers ("uhm", "chale"), and disfluencies. ~100k transcriptions.
2. **Ghana NLP Pristine-Twi (Formal):** Clean, structured, grammatically correct Twi text. ~999k samples — eliminates the TTS data imbalance that plagued the original WAXAL-only approach.

---

## 4. Experimental Groups

| Group | Training Sequence | Rationale |
|:---|:---|:---|
| **Control** | Standard Llama-3.2-1B | Baseline "Taxed" performance |
| **Variant A** | ASR Only | Pure robustness to conversational noise |
| **Variant B** | TTS Only | Maximum semantic density and logic |
| **Variant C** | ASR + TTS (Mixed) | Standard joint-distribution training |
| **Variant D** | **TTS → ASR → TTS** | **Primary hypothesis:** Anchor logic, adapt to noise, refine logic |
| **Variant E** | ASR → TTS | Test if phonetic grounding aids later reasoning |

---

## 5. Directory Structure

```
somax/
├── data/                  # Twi datasets (WAXAL ASR + Ghana NLP pristine-twi) — gitignored
├── models/                # Trained tokenizers, routers, GGUF files — gitignored
├── checkpoints/           # LoRA training checkpoints — gitignored
├── scripts/               # Research pipeline
│   ├── download.py        # Dataset downloader (google/WaxalNLP via HuggingFace)
│   ├── train_bpe.py       # Unified 8k BPE vocabulary generation
│   ├── train_router.py    # TF-IDF + logistic regression router training
│   ├── train_lora.py      # Staged LoRA training (all variants A–E)
│   ├── export_gguf.py     # LoRA merge + llama.cpp GGUF quantization
│   ├── benchmark_fertility.py  # Token fertility auditing (F = tokens/words)
│   └── benchmark_inference.py  # Edge latency / TPS / memory auditing
├── somax/                 # Edge Python library
│   ├── __init__.py        # Exports WAXALRouter, DualCoreTokenizer
│   ├── router.py          # Stream classifier (trained TF-IDF or regex fallback)
│   └── tokenizer.py       # Dual-core stream manager (unified vocabulary)
├── configs/
│   └── variants.yaml      # LoRA variant definitions (Control, A–E)
├── tests/
│   ├── test_router.py
│   └── test_tokenizer.py
├── notebooks/
│   └── train_eval.ipynb  # End-to-end Colab pipeline
├── Makefile               # Pipeline shortcuts
└── project.md             # This file
```

---

## 6. Language File Naming Convention

All scripts use a shared prefix mapping that mirrors `download.py`:

| Language | ASR prefix | Formal text prefix |
|----------|------------|--------------------|
| twi      | `aka_asr`  | `pristine_twi`     |

Files follow the pattern `{prefix}_{split}.jsonl` (e.g. `aka_asr_train.jsonl`, `pristine_twi_test.jsonl`).

---

## 7. Phase I — Vocabulary & LoRA (Cloud)

### 7.1 Dataset Download (`download.py`)

```bash
python scripts/download.py --output data/
```

Downloads two sources:
- **WAXAL `aka_asr`** from `google/WaxalNLP` — spontaneous Twi/Akan speech transcriptions
- **Pristine-Twi** from `ghananlpcommunity/pristine-twi` — 999k clean formal Twi text, auto-split 90/5/5 into train/validation/test

All saved as JSONL under `data/twi/`.

### 7.2 BPE Tokenizer (`train_bpe.py`)

Trains a single unified 8k BPE vocabulary on combined ASR+TTS data.

Special tokens (in ID order):

```
[PAD]=0  [UNK]=1  [CLS]=2  [SEP]=3  [MASK]=4  <s>=5  </s>=6  <pad>=7
```

Outputs:
- `models/tokenizers/{lang}/unified_tokenizer.json` — raw BPE (tokenizers library format)
- `models/tokenizers/{lang}/tokenizer_config.json` — bos/eos/pad mappings for `PreTrainedTokenizerFast`
- `models/tokenizers/{lang}/stream_token_stats.json` — per-token ASR/TTS dominance metadata

```bash
python scripts/train_bpe.py --input data/twi/ --output models/tokenizers/ --language twi
```

### 7.3 Router Training (`train_router.py`)

Trains a TF-IDF + logistic regression classifier on WAXAL ASR/TTS splits. Character n-grams (2–4), max 20k features. 5-fold cross-validation reported. Saved as `models/router/{lang}_router.pkl`.

```bash
python scripts/train_router.py --data data/twi/ --output models/router/ --language twi
```

### 7.4 Staged LoRA Training (`train_lora.py`)

When `--tokenizer-path` is provided, the WAXAL tokenizer replaces Llama's 128k tokenizer:

1. Load Llama-3.2-1B and **snapshot** its full 128k embedding matrix (before resize)
2. Resize model embeddings from 128k → 8k
3. **Warm-initialize** each of the 8k rows by averaging the Llama subword embeddings for that token string; fall back to the embedding mean for empty encodings
4. Wrap with LoRA using `modules_to_save=["embed_tokens", "lm_head"]` so the embedding and output projection layers stay fully trainable alongside LoRA adapters

This ensures fertility gains measured in benchmarks reflect real inference behaviour.

```bash
# Recommended — uses WAXAL tokenizer
python scripts/train_lora.py \
    --group D \
    --data data/twi/ \
    --output checkpoints/ \
    --tokenizer-path models/tokenizers/twi/unified_tokenizer.json

# Control group — uses Llama tokenizer directly
python scripts/train_lora.py --group control --data data/twi/ --output checkpoints/
```

Variant D staged training sequence:

| Stage | Data | LR | Epochs |
|-------|------|----|--------|
| 1 | TTS (formal) | 2e-4 | 2 |
| 2 | ASR (conversational) | 1e-4 | 1 |
| 3 | TTS (formal) | 5e-5 | 1 |

---

## 8. Phase II — Edge Library

### 8.1 WAXALRouter (`somax/router.py`)

Loads a trained `.pkl` classifier when available; falls back to a regex heuristic (6 conversational markers + length < 5 words) when no model file exists.

```python
router = WAXALRouter(language="twi", model_dir="models/router/")
router.classify("uhm chale me dwo")                          # → "robust"
router.classify("The president delivered a formal address")  # → "logic"
```

### 8.2 DualCoreTokenizer (`somax/tokenizer.py`)

Wraps the unified WAXAL `PreTrainedTokenizerFast` with the `WAXALRouter` for stream classification. Special tokens (`<s>`, `</s>`, `<pad>`) are set at load time.

```python
from somax import DualCoreTokenizer

tokenizer = DualCoreTokenizer(
    tokenizer_path="models/tokenizers/twi/unified_tokenizer.json",
    language="twi",
)

tokenizer.classify("uhm chale me dwo o")           # → "robust"
tokenizer.encode("The formal text goes here")      # → [token IDs]
ids, stream = tokenizer.encode_with_stream("uhm")  # → ([...], "robust")
```

---

## 9. Phase III — GGUF Export (`export_gguf.py`)

Merges LoRA adapters into the base model, then converts to GGUF using llama.cpp. Requires llama.cpp built from source in a sibling directory or on PATH.

```bash
python scripts/export_gguf.py \
    --checkpoint checkpoints/variant_D/final/ \
    --output models/gguf/ \
    --quantization Q4_K_M
```

---

## 10. Phase IV — Benchmarking

### Token Fertility

```bash
python scripts/benchmark_fertility.py \
    --tokenizer meta-llama/Llama-3.2-1B \
    --waxal-tokenizer models/tokenizers/twi/unified_tokenizer.json \
    --test-file data/twi/pristine_twi_test.jsonl \
    --compare
```

Target: ≥30% fertility reduction. Expected: baseline ~4.0 → WAXAL ~2.8 tokens/word.

### Edge Inference (Dell Latitude 7400)

```bash
python scripts/benchmark_inference.py \
    --model models/gguf/model-Q4_K_M.gguf \
    --test-file data/twi/pristine_twi_test.jsonl
```

Measures tokens/second, latency (mean ± std), and memory usage (MB).

---

## 11. Dependency Manifest

| Group | Packages |
|-------|----------|
| Core (always) | `transformers`, `tokenizers`, `psutil` |
| Train (cloud) | `peft`, `bitsandbytes`, `datasets`, `accelerate`, `torch`, `sentencepiece` |
| Edge (local inference) | `llama-cpp-python` |
| Dev | `pytest`, `black`, `ruff`, `mypy` |

```bash
pip install -e ".[dev,train]"        # cloud training + dev
pip install -e ".[edge]"             # edge inference (requires C++ compiler on Windows)
```

---

## 12. Phase V — Twi QA Evaluation

Fertility reduction is a compression metric. A Twi QA evaluation closes the gap to task performance, demonstrating that the improvements are linguistically real — not just a tokenization artifact.

**What it would demonstrate:**
- The fertility reduction produces a model that actually understands and generates Twi better
- Variant D's staged training (TTS→ASR→TTS) outperforms the control and simpler variants on a real task, not just on the compression metric
- The warm embedding initialization worked — the model converged to something meaningful despite starting from a resized vocabulary

**Dataset options:**
- **AfriQA benchmark** — covers several African languages including Akan/Twi
- **Ghana NLP community datasets**
- A hand-curated set of 100–200 Twi QA pairs is sufficient to show a clear trend across variants

**Target comparison table:**

| Model | Fertility (F) | Exact Match | F1 |
|---|---|---|---|
| Control (Llama base) | ~4.0 | baseline | baseline |
| Variant D + WAXAL tokenizer | ~2.8 | +X% | +X% |

**Honest caveat:** Given the T4 training budget and the fact that embeddings are effectively retrained from warm-initialized scratch for 8k tokens, Variant D may not beat the control on QA out of the box. If it doesn't, that is still a legitimate research finding — it quantifies how much additional training is needed for fertility gains to materialize as task gains, and establishes a clear direction for future work.

---

## 13. Roadmap

- ✅ Phase I — Baseline fertility audit
- ✅ Phase II — Staged LoRA training with WAXAL tokenizer and warm embedding init
- ✅ Phase III — GGUF export pipeline
- ⬜ Phase IV — Hardware benchmarking on Dell Latitude 7400 + GitHub release
- ⬜ Phase V — Twi QA evaluation (AfriQA or curated set, Exact Match + F1 across variants)
