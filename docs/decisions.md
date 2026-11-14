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