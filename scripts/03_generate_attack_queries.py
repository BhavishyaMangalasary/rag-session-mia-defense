"""Step 2c: write the attacker's 5 queries for each of the 100 target documents.

Output: data/attack_queries.jsonl  (one line per target: id, label, summary, questions, queries)
Safe to stop and re-run: documents already done are skipped.
"""
import json

from tqdm import tqdm

from ragmia import config as C
from ragmia.attack import attack_query, write_queries

out_path = C.DATA_DIR / "attack_queries.jsonl"
done = set()
if out_path.exists():
    done = {json.loads(line)["id"] for line in out_path.open() if line.strip()}

targets = [json.loads(line) for line in (C.DATA_DIR / "targets.jsonl").open()]
todo = [t for t in targets if t["id"] not in done]
print(f"{len(done)} already done, {len(todo)} to go (query model: {C.QUERY_MODEL})")

with out_path.open("a") as f:
    for t in tqdm(todo):
        summary, questions = write_queries(t)
        f.write(json.dumps({
            "id": t["id"], "label": t["label"], "summary": summary, "questions": questions,
            "queries": [attack_query(summary, q) for q in questions],
        }) + "\n")
        f.flush()

example = json.loads(out_path.open().readline())
print(f"\nExample ({example['id']}, label={example['label']}):")
for q in example["queries"]:
    print(" -", q)