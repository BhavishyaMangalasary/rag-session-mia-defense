# Session-Level Defense Against Membership Inference on RAG Systems

Reproduction of **MEntA** (Nguyen et al., USENIX Security 2026) and a new
**session-level extension of the Mirabel defense**.
CS 528 Data Security and Privacy · Illinois Institute of Technology · Fall 2026.

## In plain terms

A RAG assistant answers questions from a private document collection. MEntA shows
that an attacker holding a copy of a document can find out whether it is in that
collection by asking just **5 normal-looking questions** and checking whether the
answers contain facts from the document.

The strongest existing defense, Mirabel, judges each question on its own. It catches
the attack, but on corpora like NFCorpus and SCIDOCS it also flags **60–88% of honest
queries**, because honest users ask focused questions too.

This project tests a different signal: **repetition within a session**. An honest
user may ask one sharp question, but an attacker's questions keep landing on the
**same document**. The defense watches a rolling window of recent queries and hides a
document from the context once it is the top match too many times.

## Threat model

- **Target:** black-box RAG. The attacker sends text and reads text only.
- **Goal:** decide whether a candidate document is in the private store.
- **Attacker has:** a copy of the document, a few known non-members for calibration,
  and a small query budget.
- **Defender has:** retrieval results and similarity scores for every query, grouped
  by session.

## Experiment

Three conditions, each evaluated on attack sessions and honest sessions:

| Condition | Description |
|---|---|
| A. No defense | Reproduces MEntA's attack |
| B. Mirabel | Per-query similarity-outlier detection + hide |
| C. Session-level (ours) | Flag when one doc is top-1 in ≥K of the last N queries + hide |

**Metrics:** attack AUC and TPR@1%FPR · share of attack sessions flagged ·
share of honest sessions flagged (false-positive rate) · honest-query utility.

## Repository layout

```
ragmia/       library code (config, data, RAG system; attack and defenses to come)
scripts/      numbered pipeline steps, run in order
docs/         design decisions, report, slides
data/         generated locally by scripts/01 (not committed)
```

## Setup

Requires macOS or Linux, Python 3.12, and [Ollama](https://ollama.com).

```bash
git clone https://github.com/BhavishyaMangalasary/rag-session-mia-defense.git
cd rag-session-mia-defense
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
ollama pull qwen2.5:7b
```

**macOS note:** if `import ragmia` fails with `ModuleNotFoundError` after installing,
macOS may have marked the environment's files as hidden (common for folders synced to
iCloud, such as Documents). Fix with `chflags -R nohidden .venv`, or keep the project
outside iCloud-synced folders.

## Running

| Step | Command | Status |
|---|---|---|
| 1. Prepare data and index | `python scripts/01_prepare_data.py` | ✅ |
| 1. Smoke-test the RAG | `python scripts/02_smoke_test.py --generate` | ✅ |
| 2. Generate attacker queries | `python scripts/03_generate_attack_queries.py` | ✅ |
| 2. MEntA attack (condition A) | `python scripts/04_run_attack.py` | ✅ |
| 3. Mirabel baseline (condition B) | — | ⏳ |
| 4. Session-level defense (condition C) | — | ⏳ |
| 5. Benign and adaptive sessions | — | ⏳ |
| 6. Full evaluation and plots | — | ⏳ |

Design choices and deviations from the paper are logged in
[`docs/decisions.md`](docs/decisions.md).

## Results

*To be filled in after the runs.*

| Condition | Attack AUC | TPR@1%FPR | Attack sessions flagged | Honest sessions flagged |
|---|---|---|---|---|
| A. No defense | 0.986 (95% CI 0.967–0.999) | 0.66 | n/a | n/a |
| B. Mirabel | – | – | – | – |
| C. Session-level | – | – | – | – |

## Limitations

- Small sample (50 + 50 documents), so results are noisier than in the paper.
- Answers returned before a session is flagged still reach the attacker.
- An attacker who spreads queries across sessions or identities is not seen by a
  per-session monitor; this is evaluated with an adaptive attacker.

## Ethics

Only public benchmark data (BEIR) is used. No deployed system is attacked. The
attack is already published; this work focuses on the defense.

## References

1. Nguyen et al. *Five Queries Are Enough: Query-Efficient and Surrogate-Free
   Membership Inference Attacks on RAG via Entailment.* USENIX Security 2026.
   Artifact: https://doi.org/10.5281/zenodo.20335970
2. Choi et al. *Safeguarding Privacy of Retrieval Data against Membership Inference
   Attacks: Is This Query Too Close to Home?* Findings of EMNLP 2025.
3. Naseh et al. *Riddle Me This! Stealthy Membership Inference for
   Retrieval-Augmented Generation.* ACM CCS 2025.
4. Thakur et al. *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of
   Information Retrieval Models.* NeurIPS Datasets & Benchmarks 2021.