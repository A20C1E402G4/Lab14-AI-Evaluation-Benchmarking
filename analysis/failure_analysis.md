# Failure Analysis Report — Lab 14 AI Evaluation Factory

## 1. Benchmark Overview

| Metric | V1 (top_k=3) | V2 (top_k=5) |
|--------|-------------|-------------|
| Total cases | 70 | 70 |
| Pass (score ≥ 3) | 57 (81.4%) | 58 (82.9%) |
| Fail (score < 3) | 13 (18.6%) | 12 (17.1%) |
| Avg LLM-Judge score | 4.05 / 5.0 | 4.06 / 5.0 |
| Hit Rate (top-3) | 74.3% | 74.3% |
| MRR | 0.648 | 0.657 |
| Agreement Rate (Claude+GPT) | 88.6% | 88.6% |

**Score distribution (V2):** 1.0×7, 2.0×5, 3.0×1, 3.5×1, 4.0×15, 4.5×8, 5.0×33

---

## 2. Failure Clustering

| Cluster | Count | Description |
|---------|-------|-------------|
| Retrieval miss — BM25 failure | 5 | Question uses different vocabulary than chunk text (e.g., "normal equations" but chunk uses "closed-form solution") |
| Context truncation | 4 | Retrieved context was cut at 1500 chars; formula/derivation needed was in the truncated portion |
| Out-of-context / adversarial | 3 | Agent correctly declined to answer but got scored low by judges expecting a technical response |

---

## 3. Five Whys — Three Worst Cases

### Case #1: "What is the normal equations method used for in linear regression?" (Score 1.0)

**Symptom:** Agent responded "I don't have information about this topic" despite the answer being in the chunks.

1. **Why 1:** Agent returned "I don't have information…" → context was not retrieved.
2. **Why 2:** BM25 retrieved chunks about "gradient descent" instead of "normal equations".
3. **Why 3:** The phrase "normal equations" does not appear verbatim in the chunks — they say "closed-form solution" or "matrix derivative".
4. **Why 4:** BM25 is a term-frequency model; it fails on vocabulary mismatch between query and document.
5. **Root Cause:** Keyword-based BM25 retrieval has no semantic understanding. A semantic/embedding-based retriever (e.g., OpenAI embeddings + cosine similarity) would match synonymous phrases correctly.

**Fix:** Replace BM25 with a hybrid sparse+dense retriever using OpenAI `text-embedding-3-large` for the dense component.

---

### Case #2: "Explain why trAB = trBA is useful for simplifying matrix derivative expressions" (Score 1.0)

**Symptom:** Agent said "I don't have information" even though the matrix trace property appears in the chunk AND hit_rate=1.0 (BM25 retrieved the right chunk).

1. **Why 1:** Agent output was "I don't have information…" despite the correct chunk being retrieved.
2. **Why 2:** The retrieved chunk text was truncated to 1500 characters; the specific trace-transpose identity was in the truncated portion.
3. **Why 3:** Chunk `ml_1_chunk_7` has 4464 characters; our `[:1500]` limit cut the critical derivation.
4. **Why 4:** Context window budget was set conservatively to keep prompt tokens low.
5. **Root Cause:** Hard truncation of long chunks destroys the very content the user asks about. The chunk should be semantically split smaller (≤ 800 chars) at generation time, or the context window limit should be raised.

**Fix:** Increase context character limit from 1500 → 3000, or apply semantic chunking to keep each chunk under 1000 chars with meaningful boundaries.

---

### Case #3: "Explain why (X·θ − y)ᵀ(X·θ − y) equals the sum of squared errors" (Score 1.0)

**Symptom:** Correct formula derivation is in the lecture notes but BM25 returned unrelated chunks.

1. **Why 1:** Agent answered with "I don't have information."
2. **Why 2:** BM25 scored chunks about Newton's method higher than the quadratic cost derivation chunk.
3. **Why 3:** Query has LaTeX notation `(X*theta - y)^T`; chunks contain the formula in rendered Unicode `(Xθ − y)ᵀ`.
4. **Why 4:** BM25 tokenizes on whitespace; `(X*theta` and `(Xθ` are different tokens.
5. **Root Cause:** LaTeX ↔ Unicode formula mismatch is invisible to a bag-of-words model.

**Fix:** Normalize all mathematical notation in both queries and chunks to a canonical plain-English form during preprocessing (e.g., "X theta minus y transposed times X theta minus y").

---

## 4. Action Plan

- [ ] **Dense retrieval:** Replace BM25-only with hybrid BM25 + OpenAI embeddings (cosine) retrieval for better semantic matching.
- [ ] **Chunk size:** Reduce maximum chunk size from current (up to 6000 chars) to ≤ 1000 chars with overlap; re-run `synthetic_gen.py`.
- [ ] **Formula normalization:** Pre-process chunks and queries to normalize LaTeX → plain-text tokens.
- [ ] **Context budget:** Increase per-chunk context limit from 1500 → 3000 chars in the agent prompt.
- [ ] **Judge calibration:** The 7 cases with score=1.0 all received the same score from both Claude and GPT; investigate if the rubric is too binary for edge cases.
