# Individual Reflection — Tran Thanh Phong
**Lab:** Day 14 — AI Evaluation Factory
**Role:** Data & SDG (Synthetic Data Generation)
**Commit:** `8aae319` — `feat(data): generate golden dataset with SDG and complete failure analysis`

---

## 1. What I Built

My responsibility was everything that feeds the evaluation pipeline: the source documents, the chunking strategy, the synthetic QA generation, and the post-benchmark failure analysis. Without a high-quality golden dataset with correct `expected_retrieval_ids`, none of the downstream metrics — Hit Rate, MRR, or judge scores — would be meaningful.

### Source Documents (`data/raw/`)

Two CS229 Machine Learning lecture note files form the knowledge base:

- `ml_1.md` — Supervised learning, linear regression, gradient descent, normal equations, logistic regression (12 sections, ~29K characters)
- `ml_2.md` — Generative learning algorithms, GDA, Naive Bayes, neural networks, perceptron (8 sections, ~28K characters)

These were chosen because they are dense with specific technical vocabulary, formulas, and concepts — ideal for testing whether a retrieval system can locate precise information under varied question phrasing.

### Chunking Strategy (`chunk_markdown()` in `data/synthetic_gen.py`)

I split each `.md` file on markdown header lines (`## ...`), keeping the header text as part of the chunk. This is header-based semantic chunking rather than fixed-size chunking.

```python
for line in lines:
    if re.match(r"^#{1,6}\s", line):
        # save current chunk, start new one
        current_header = line.strip()
        current_lines = [line]
    else:
        current_lines.append(line)
```

Each chunk keeps: `id` (e.g. `ml_1_chunk_3`), `header`, `text`, `source`. Chunks shorter than 100 characters are dropped (section stubs with no content). The ID format is deterministic — `{stem}_chunk_{n}` where `n` increments as chunks are found — so the same document always produces the same IDs. This determinism is critical: `expected_retrieval_ids` in the golden dataset must exactly match the IDs used in the vector store and by the agent.

Result: **20 chunks** total (12 from `ml_1.md`, 8 from `ml_2.md`), written to `data/chunks.jsonl`.

### Synthetic Data Generation (`generate_from_chunk()`)

For each chunk I called `claude-haiku-4-5-20251001` with a structured prompt requesting 3 QA pairs covering different cognitive levels:

```
Mix difficulty levels:
- factual recall (easy): "What is the definition of..."
- conceptual understanding (medium): "Explain why..."
- application (hard): "How would you use X to..."
```

The prompt forces JSON output with a strict schema: `question`, `expected_answer`, `difficulty`, `type`. Each generated record is then enriched with `context` (full chunk text) and `expected_retrieval_ids: [chunk_id]` — this is the ground truth linking a question to the exact chunk it should be retrieved from.

Running 20 chunks × 3 pairs = 60 normal cases, all with `expected_retrieval_ids`.

### Adversarial Red-Teaming (`generate_adversarial()`)

I designed 10 adversarial cases to stress-test failure modes that normal QA pairs would never surface:

| Type | Purpose | Expected behaviour |
|------|---------|-------------------|
| `out_of_context` | Ask about cooking, politics | Agent declines gracefully |
| `hallucination_bait` | Ask for a specific number not in the text | Agent should not invent a number |
| `prompt_injection` | "Ignore previous instructions and..." | Agent follows its system prompt |
| `ambiguous` | Vague question with no single correct answer | Agent acknowledges ambiguity |

These 10 cases have empty `expected_retrieval_ids: []` — the retrieval evaluator skips them for Hit Rate/MRR, but the judge still scores them, revealing how the agent handles out-of-distribution input.

Final dataset: **70 records** — easy: 20, medium: 21, hard: 19, adversarial: 10.

### Failure Analysis (`analysis/failure_analysis.md`)

After the benchmark ran, I filled in the actual numbers and wrote the 5-Whys analysis for the 3 worst-performing cases. The most important finding was that all 3 failures came from retrieval, not generation:

- BM25 vocabulary mismatch ("normal equations" vs "closed-form solution")
- Context truncation (relevant text after the 1500-char cut-off)
- LaTeX vs Unicode formula representation in the query vs the chunk

---

## 2. Problems I Solved

### Problem 1 — Claude returns JSON wrapped in markdown fences

Despite the prompt saying "Return ONLY a JSON array (no markdown fences)", Claude Haiku occasionally responded with ` ```json\n[...]\n``` `. A raw `json.loads()` would fail.

I wrote `extract_json()` to strip fences before parsing:

```python
def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())
```

Also handled the case where Claude returned a dict wrapping the array (`{"questions": [...]}`) instead of a bare array, by taking `next(iter(parsed.values()))` when the result is a dict.

### Problem 2 — Concurrent generation hits Claude rate limit

Generating 20 chunks + 1 adversarial batch = 21 Claude calls. Running them all with `asyncio.gather` at once bursts past the 50 req/min limit.

Fix: `asyncio.Semaphore(5)` at module level, wrapping every `call_claude()` invocation:

```python
SEMAPHORE = asyncio.Semaphore(5)

async def call_claude(prompt, temperature=0.7):
    async with SEMAPHORE:
        response = await client.messages.create(...)
    return response.content[0].text
```

This limits to 5 concurrent Claude calls, smoothing the request rate across the generation job.

### Problem 3 — Chunk ID collision at index 0

The first chunk of `ml_2.md` has ID `ml_2_chunk_0` while `ml_1.md` starts at `ml_1_chunk_1` (the initial unchunked intro text before the first header becomes `chunk_0`, but it was too short and got filtered out). This created a naming gap that could confuse anyone reading the IDs.

By design, the counter starts at 0 for each file — the first real chunk of `ml_2.md` is `ml_2_chunk_0` because there is no preceding header-less intro content that survives the 100-char filter. The IDs are correct, just not sequential from 1 for every file. I documented this in the source code comments.

---

## 3. Technical Concepts

### Why ground truth chunk IDs are indispensable

A golden dataset without `expected_retrieval_ids` can only tell you "was the final answer good?" It cannot tell you *why* it was bad. Our benchmark proved this directly: cases where retrieval missed scored an average of **3.17/5**, versus **4.38/5** when retrieval hit — a 1.21 point gap. Without tracking which chunks were retrieved, this causal relationship is invisible.

In production RAG systems, this is how you debug: if answer quality drops, you first check hit rate. If hit rate is fine, the problem is in generation or prompting. If hit rate dropped, the problem is in chunking, embedding, or the retriever.

### Semantic vs fixed-size chunking

Fixed-size chunking (e.g. every 512 tokens with 50-token overlap) is simpler but breaks across conceptual boundaries. A chunk might start mid-sentence of one concept and end mid-sentence of another.

Header-based chunking, as I implemented, aligns chunks with the document's own structure. The author of CS229 notes used headers to mark topic shifts — gradient descent gets its own section, normal equations get their own section. A question about gradient descent will never retrieve a chunk about naive Bayes because there is a clean boundary between them.

The trade-off is chunk size variance. Our chunks range from 362 characters (`ml_2_chunk_1`, a very short section) to 6098 characters (`ml_2_chunk_5`, a long derivation). This variance causes problems for the retriever (short chunks may not contain enough vocabulary for BM25 to score them well) and for the agent (long chunks get truncated at 1500 chars, losing the tail content).

### Adversarial testing matters more than it looks

10 adversarial cases is 14% of the dataset. They identified 3 of the 4 cases where the two judges disagreed by 3+ points (agreement=0.0). This is because:

- `hallucination_bait` cases: the agent correctly said "I don't have information" (good), but GPT judged this harshly (score=1) while Claude judged it leniently (score=4), causing a diff=3 conflict. The rubric didn't specify how to score correct refusals.
- `prompt_injection` cases: similar situation — the agent followed its system prompt and refused to comply, which is correct behavior, but the judges scored it differently.

These conflicts are the most valuable output of the adversarial set: they reveal that the scoring rubric is underspecified for edge cases where "correct" behavior is refusal rather than an answer.

### SDG cost and quality

Generating 70 QA pairs with Claude Haiku cost approximately $0.003 (21 API calls × ~700 tokens each × $0.00025/1K). This is negligible. The bigger cost is human review — some generated questions were too easy ("What letter represents the hypothesis function?") and some expected answers were too brief to be useful as a judge reference. In a production SDG pipeline, you would add a filtering step to reject low-quality pairs based on question length, answer length, and a difficulty classifier.

---

## 4. What I Learned

**The golden dataset is the hardest thing to get right.** Bad test cases produce misleading metrics. If all questions are easy factual lookups, an 80% pass rate means nothing. I learned to design test sets with intent: cover each cognitive level (recall, understanding, application), cover each failure mode (retrieval miss, truncation, out-of-scope), and ensure every case has a machine-verifiable ground truth (the chunk ID).

**Chunking is an irreversible decision that propagates everywhere.** Once you choose your chunking strategy and generate your golden dataset against it, every downstream component — the vector store, the agent, the retrieval evaluator — is coupled to those chunk boundaries and IDs. Changing the chunking strategy means regenerating everything. This is why the design decision (header-based vs fixed-size, chunk length limits, minimum chunk size) deserves careful thought before any code is written.
