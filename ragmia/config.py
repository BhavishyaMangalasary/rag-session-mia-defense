from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# Dataset (BEIR format: corpus.jsonl, queries.jsonl, qrels/*.tsv)
DATASET = "nfcorpus"
BEIR_URL = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"

# Member / non-member sampling
SEED = 528
N_MEMBERS = 50
N_NONMEMBERS = 50
MIN_WORDS = 80             # skip very short docs (too little to ask 5 questions about)
MIN_QUERIES_PER_DOC = 2    # target docs need >=2 real user queries (used for benign sessions in Step 5)
DUP_SIM_THRESHOLD = 0.95   # a non-member with a store doc above this cosine sim is a near-duplicate -> rejected

# RAG system
RETRIEVER_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 3
OLLAMA_URL = "http://localhost:11434/api/chat"
GENERATOR_MODEL = "llama3.1:8b"
GEN_TEMPERATURE = 0.0
GEN_MAX_TOKENS = 400

# Attack (MEntA)
QUERY_MODEL = "qwen2.5:7b"        # writes the attacker's questions; must differ from GENERATOR_MODEL
NUM_QUERIES = 5                   # query budget per target document (paper default)
NLI_MODEL = "tasksource/deberta-base-long-nli"   # the "fact checker" (same model as the paper)
NLI_MAX_LEN = 1024                # tokens; NFCorpus abstracts fit comfortably
RESULTS_DIR = ROOT / "results"


# Defenses
MIRABEL_RHO = 0.05                # significance level (Mirabel's recommended value, also used by MEntA)