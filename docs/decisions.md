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


## Step 3 — Mirabel baseline (condition B)

| Choice | Value | Why |
|---|---|---|
| Detection rule | Choi et al. (EMNLP Findings 2025), Algorithm 1: flag if s_max > τ, τ = μ_q + σ_q·√(2 ln n) + c·σ_q/√(2 ln n), c = −ln(−ln(1−ρ)) | Faithful to the published algorithm |
| μ_q, σ_q | Mean and std of the query's similarity to every store document except the top one (n = 3,582) | As specified in Mirabel |
| ρ | 0.05 | Mirabel's recommended value; also MEntA's setting |
| Embedder | `all-mpnet-base-v2`, the RAG's own retriever **[deviation]** | Mirabel's paper used BGE-m3. Here the defense sits inside the RAG and uses its retriever's scores |
| Hide action | Remove the flagged top-1 document; the next-ranked document fills the slot | Mirabel's detect-and-hide. Its paper does not say whether the slot is back-filled; we back-fill so the LLM always sees top-k documents |
| Memory | None; every query is judged independently | Mirabel is a per-query defense |

### Detection rates (no LLM, `scripts/05_mirabel_detection.py`)

| Query set | Flagged |
|---|---|
| Honest NFCorpus queries (3,237) | **50.6%** (paper, Table 8: 59.6%) |
| MEntA queries (500) | 59.4% |
| … target is a member | 72.0% |
| … target is a non-member | 46.8% |
| MEntA sessions with ≥1 flag | 79.0% |

Judged one query at a time, honest and attack queries are flagged at similar rates. Mirabel's
signal ("one document matches unusually well") is shared by honest, specific questions.

### Attack results under Mirabel

| Metric | A. No defense | B. Mirabel |
|---|---|---|
| AUC | 0.986 (0.967–0.999) | **0.756** (0.652–0.846) |
| TPR @ 1% FPR | 0.66 | 0.22 |
| TPR @ 10% FPR | 0.96 | 0.52 |

Per member document under Mirabel (out of 5 queries): 3.60 queries flagged, target hidden
3.18 times, but the target still reached the context 1.52 times on unflagged queries, giving
2.0 entailment hits. **Mirabel weakens the attack but does not stop it, at the cost of flagging
about half of honest queries.** Because it has no memory, a document hidden on one query can
leak on the next.


## Step 4 — Session-level defenses (conditions C and D, ours)

| Choice | Value | Why |
|---|---|---|
| C. Session rule | Track the top-1 document of the last N=5 queries; once one document is top-1 in ≥K=3 of them, hide it for the rest of the session | An attacker's queries keep landing on the same document; one specific honest question does not repeat. N and K from the proposal |
| Counting | The raw top-1 is counted even after it is hidden | Otherwise hiding would reset the counter and the attacker could escape the rule |
| Session | One target document's 5 attack queries, with a fresh defense object per session | Matches how MEntA is run per target |
| D. Combined | Mirabel's per-query flag + session memory (any flagged document stays hidden for the rest of the session) + the C rule | Step 3 showed Mirabel hides the target on flagged queries but it leaks again on the next unflagged one |
| Tests | 10 unit tests in `tests/`, run on every push by GitHub Actions | Defense logic needs no models, so it is checked automatically |

### Attack results (all conditions)

| Metric | A. None | B. Mirabel | C. Session | D. Mirabel + memory |
|---|---|---|---|---|
| AUC (95% CI) | 0.986 (0.967–0.999) | 0.756 (0.652–0.846) | 0.939 (0.887–0.982) | **0.669** (0.563–0.775) |
| TPR @ 1% FPR | 0.66 | 0.22 | 0.18 | **0.14** |
| TPR @ 10% FPR | 0.96 | 0.52 | 0.56 | **0.44** |
| Attack sessions flagged | — | 79% | 82% | 93% |
| Target in context (members, of 5) | 4.68 | 1.52 | 2.42 | **0.78** |
| Queries flagged (non-member sessions, of 5) | — | 2.34 | 1.64 | 3.26 |

### Key finding: one answer is almost enough (`scripts/06_query_budget.py`)

| Answers used | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| AUC, no defense | **0.924** | 0.971 | 0.983 | 0.987 | 0.986 |
| AUC, session (C) | 0.924 | 0.971 | 0.960 | 0.944 | 0.939 |
| AUC, Mirabel (B) | 0.662 | 0.744 | 0.745 | 0.742 | 0.756 |

A single MEntA answer already separates members from non-members with AUC 0.924. Any defense
that waits to observe a pattern (C, at any K) lets at least the first answer through, so it
cannot push the attacker's AUC below about 0.92. Only a defense that can act on the first
query (B, D) lowers it further, and such a defense must judge each query alone, which is
what raises false positives on honest users. With 1–2 answers the attacker's TPR @ 1% FPR is
0.00 (scores are too coarse), so strict operating points stay protected early in a session.

**Interpretation.** D gives the strongest protection (AUC 0.669), but its confidence interval
overlaps with B's, so with 100 documents the improvement is suggestive rather than proven.
D also hides more often (3.26 of 5 queries even in non-member sessions), which points to a
higher cost for honest users. Step 5 measures that cost for every condition.


## Step 5 — Cost to honest users and robustness to adaptive attackers

All of Step 5 replays retrieval only (no LLM): whether a defense hides a document depends only on
similarity scores. The replayed attack flag rates match the full Step 4 runs exactly
(79% / 82% / 93%; target leaks 4.68 / 1.52 / 2.42 / 0.78), which validates the replay.

| Choice | Value | Why |
|---|---|---|
| Honest "focused" sessions | 200 × 5 real NFCorpus queries, all judged relevant to the same store document (seed 528) | A real user digging into one topic: the case closest to an attack |
| Honest "browsing" sessions | 200 × 5 random real NFCorpus queries | Everyday traffic |
| "Flagged" | The defense hid at least one document during the session | Same definition as for attack sessions |
| Utility cost | Share of queries whose top-3 changed; relevant documents (NFCorpus judgments) lost from the top-3 | What an honest user actually loses |
| Honest "deep-dive" sessions | 100 × 5 follow-up questions written by Qwen from the paper's **title** only | Stress test: a researcher focusing on one specific paper |
| "Reader" sessions | MEntA's 5 questions without the "I am inquiring about" prefix | Someone who has the document; indistinguishable from an attacker by design (upper bound) |
| Adaptive attacker | 0–3 random real decoy queries after each attack question; with and without prefix | Tries to keep the target from repeating inside the defense's window |

### Honest users (real NFCorpus sessions)

| | B. Mirabel | C. Session (5,3) | D. Mirabel + memory |
|---|---|---|---|
| Attack sessions flagged | 79% | 82% | 93% |
| Focused honest sessions flagged | 95.5% | **0%** | 95.5% |
| Browsing honest sessions flagged | 97.0% | **0%** | 97.0% |
| Honest queries with changed context | ~50% | **0%** | ~50% |
| Relevant docs lost per query | 0.07–0.10 | **0** | 0.07–0.10 |

Session rule sweep (selected): (N=3, K=2) flags 98% of attacks, 6% of focused and 0.5% of
browsing honest sessions; (N=5, K=2) flags 99% / 8.5% / 0.5%. Full table: `results/session_sweep.csv`.

### Harder honest users (simulated)

| Sessions flagged | Mirabel | Session (5,3) | Mirabel + memory | Session (3,2) | Session (5,2) |
|---|---|---|---|---|---|
| Attack (MEntA) | 79% | 82% | 93% | 98% | 99% |
| Honest deep-dive | 75% | 70% | 86% | 86% | 90% |
| Reader (no prefix) | 73% | 57% | 83% | 87% | 96% |

Because the deep-dive prompt sees the paper's exact title, its questions target that specific
study, which is what an attacker does too. **No retrieval-based defense can tell an attacker
from an honest user who wants the same specific document: intent is not visible in the query
stream.** The session defense's advantage is over everyday traffic, not over paper-specific
deep dives. Dropping the prefix lowers the session defense's detection (82% → 57%).

### Adaptive attacker: target leaks per member session (of 5 attack questions)

| Defense | 0 decoys | 1 | 2 | 3 |
|---|---|---|---|---|
| None | 4.68 | 4.68 | 4.68 | 4.68 |
| Mirabel | 1.52 | 1.52 | 1.52 | 1.52 |
| Session (5,3) | 2.42 | 2.56 | **4.68** | 4.68 |
| Session (3,2) | 1.46 | 1.52 | **4.68** | 4.68 |
| Session (5,2) | 1.42 | 1.46 | 1.52 | 1.52 |
| Mirabel + memory | **0.78** | **0.82** | **0.94** | **0.94** |

With 2 decoys per question (15 queries instead of 5) the attacker fully evades the session rule
at (5,3) and (3,2). (5,2) resists up to 3 decoys, but any fixed window is evaded once decoys
spread the target further apart than the window, and an attacker who uses separate sessions or
accounts is invisible to any per-session defense. The combined defense D stays below one leak
per session even with decoys, because Mirabel's per-query test is not affected by spacing.
Without the prefix the attacker gets the target less often (3.98 vs 4.68 undefended) but every
defense also hides it less (see `results/adaptive_attacker.csv`).

### Overall conclusion

| | Mirabel | Session (ours) | Mirabel + memory (ours) |
|---|---|---|---|
| Detects attacks | 79% | 82% (98% at K=2) | **93%** |
| Bothers everyday honest users | ~96% | **0%** | ~96% |
| Limits leakage (attacker AUC) | 0.756 | 0.939 | **0.669** |
| Robust to decoys | yes | no | **yes** |

The session rule is a near-free **detector** for everyday traffic (flag, rate-limit, review),
but on its own it cannot stop the leak (one answer already gives AUC 0.924) and decoys evade it.
Adding session memory to Mirabel gives the strongest protection against both the standard and
the adaptive attacker, at Mirabel's cost to honest users. No single defense wins on every axis.


## Step 7 — Second dataset: SCIDOCS (branch `scidocs`)

Every step was rerun unchanged on BEIR SCIDOCS (25,657 computer-science abstracts, 1,000 queries)
to test whether the NFCorpus findings generalise. Dataset is chosen per run with
`RAGMIA_DATASET=scidocs`; each dataset has its own `data/<dataset>/` and `results/<dataset>/`.

| Choice | Value | Why |
|---|---|---|
| Why SCIDOCS | Used by both MEntA and Mirabel; the paper reports Mirabel's worst benign FPR there (88.1%) | Strongest test of the honest-user claim |
| Not TREC-COVID | — | 171k documents (too slow to embed on a laptop) and only 50 queries (too few for honest sessions) |
| Target filter | Same as NFCorpus (≥80 words, ≥2 real queries): 416 candidates, 2 near-duplicates rejected, store = 25,607 | Identical design across datasets |
| Focused honest sessions | 40 (only 44 SCIDOCS documents have ≥5 real queries) **[deviation]** | Browsing (200) and deep-dive (100) sessions unchanged |
| Everything else | Same models, prompts, settings, seed | Only the corpus changes |

### Results: NFCorpus vs SCIDOCS

| Metric | NFCorpus | SCIDOCS |
|---|---|---|
| Attack AUC, no defense (paper, Table 5) | 0.986 (0.935) | 0.978 (0.936) |
| AUC from 1 answer only | 0.924 | 0.892 |
| Mirabel: single honest queries flagged (paper) | 50.6% (59.6%) | 81.4% (88.1%) |
| AUC: Mirabel / session / Mirabel + memory | 0.756 / 0.939 / 0.669 | 0.651 / 0.903 / **0.594** (CI 0.486–0.697) |
| TPR @ 1% FPR: Mirabel / session / Mirabel + memory | 0.22 / 0.18 / 0.14 | 0.10 / 0.18 / 0.04 |
| Attack sessions flagged: Mirabel / session / Mirabel + memory | 79% / 82% / 93% | 99% / 71% / 99% |
| Real honest sessions flagged, Mirabel (focused / browsing) | 95.5% / 97.0% | 97.5% / 100% |
| Real honest sessions flagged, session rule | **0% / 0%** | **0% / 0%** |
| Session rule N=5, K=2: attacks / focused honest | 99% / 8.5% | 95% / 2.5% |
| Honest deep-dive flagged: Mirabel / session | 75% / 70% | 98% / 58% |
| Target leaks per member session, Mirabel + memory (0 → 3 decoys) | 0.78 → 0.94 | 0.42 → 0.42 |
| Decoys (2 per question) evade session rule (5,3) | yes | yes |

**Interpretation.** Every finding repeats on a second domain. The defenses rank the same on both
datasets (Mirabel + memory < Mirabel < session < none), one answer is almost enough, Mirabel
flags nearly every honest session, and the session rule flags none. On SCIDOCS, Mirabel + memory
brings the attacker's AUC to 0.594 with a CI that includes 0.5. The session rule catches fewer
attacks on SCIDOCS (71% vs 82%), likely because the larger corpus has more near-identical papers,
so the same document wins top-1 less consistently; at K=2 it still catches 95%. Against harder
deep-dive users its advantage over Mirabel is larger on SCIDOCS (58% vs 98% flagged).