# Design decisions

A running log of every choice that affects results, and why. Deviations from the
MEntA paper are marked **[deviation]**.

## Step 1 — Data and RAG system

| Choice | Value | Why |
|---|---|---|
| Dataset | BEIR NFCorpus (3,633 medical abstracts, 3,237 queries) | Used in MEntA; also the corpus where Mirabel's benign FPR was reported as 0.596 |
| Sample size | 50 members + 50 non-members **[deviation]** | Paper uses 1,000 + 1,000. Scaled down to fit a laptop and a 3-week timeline; confidence intervals will be wider |
| Target pool | Docs with ≥80 words and ≥2 real NFCorpus queries (3,266 docs qualified) | Enough content for 5 questions; real queries let us build realistic benign sessions on the same docs (Step 5) |
| Member/non-member assignment | Random from the same pool (seed 528) | Prevents one group from being systematically easier |
| Near-duplicate filter | Reject a non-member if any store doc has cosine sim ≥ 0.95 (3 candidates rejected) | Otherwise the RAG can answer from the duplicate and the "non-member" label is wrong |
| RAG store | Full corpus minus the 50 non-members (3,583 docs) | Members compete with thousands of background docs, as in a real deployment |
| Retriever | `all-mpnet-base-v2`, exact cosine search | One of the paper's two retrievers. Exact search instead of FAISS **[deviation]**: identical results at this corpus size |
| Top-k | 3 | Paper's default |
| Generator | Llama 3.1 8B via Ollama, temperature 0 | One of the paper's four generators; runs locally on 16 GB Apple Silicon. Temperature 0 for reproducibility |
| System prompt | MEntA's abstention prompt (Appendix A): answer in 1–2 sentences from the context only, otherwise reply exactly "I don't know" | Matches the paper's target system. Refusals are part of MEntA's score (penalty), so the target must abstain the way the paper's does. Verified: member question → specific answer quoting the doc; non-member question → "I don't know" |

### Sanity checks (Step 1)

- Searching each member's own title returns it in the top 3: **50/50**.
- No non-member is present in the store.
- End-to-end example: a member question (MED-2217) retrieved the member and the
  answer quoted its exact statistics; a non-member question (MED-4735) was answered
  from neighbouring documents only. This is the leakage MEntA exploits.


## Step 2 — MEntA attack (condition A: no defense)

| Choice | Value | Why |
|---|---|---|
| Query writer | Qwen 2.5 7B via Ollama **[deviation]** | Must differ from the RAG generator so the target cannot "recognize" its own phrasing. Paper used an OpenAI model; Qwen runs locally for free |
| Query prompt | Paper's Figure 13 template, verbatim | Faithful reproduction; spreads 5 questions across beginning, middle and end of the document |
| Topic prefix | "I am inquiring about <topic>. <question>"; topic phrase written by the query model | Summary-prepending from IA, used by MEntA to steer retrieval toward the target. The paper does not print its summary prompt, so ours is written to produce the short topic phrase shown in its Table 2 |
| Query budget | 5 per document | Paper default |
| NLI model | `tasksource/deberta-base-long-nli` (argmax = entailment) | Same model as the paper |
| Claim splitting | Answer split into sentences / lines, bullets removed | Paper splits answers into atomic sentences before checking entailment |
| Refusal hypotheses | Paper's 7 paraphrases + "I don't know." **[deviation]** | Our target is told to reply exactly "I don't know", so the phrase itself is included |
| Score | mean over 5 queries of (hit − refusal) | Paper Eq. 4 |

### Results (condition A)

| Metric | Ours (50 + 50 docs) | Paper (Table 5: Llama 3.1 8B, all-mpnet-base-v2, NFCorpus, top-3) |
|---|---|---|
| AUC | **0.986** (95% CI 0.967–0.999) | 0.935 |
| TPR @ 1% FPR | **0.66** | 0.363 |
| TPR @ 10% FPR | **0.96** | — |

Per document, out of 5 queries:

| | Target retrieved | Hits | Refusals | Mean score |
|---|---|---|---|---|
| Members | 4.68 | 4.46 | 0.66 | +0.76 |
| Non-members | 0.00 | 0.54 | 3.98 | −0.69 |

**Interpretation.** The reproduction confirms the paper's claim: five natural queries
separate members from non-members almost perfectly. Our AUC is higher than the paper's,
likely because of (1) the small sample (wide CI), (2) the target pool restricted to
longer documents with real user queries, which are more distinctive, (3) the
near-duplicate filter, which removes the similar non-members the paper names as a main
source of false positives, and (4) temperature-0 generation and a different query writer.
With 50 non-members, TPR @ 1% FPR means zero false positives.