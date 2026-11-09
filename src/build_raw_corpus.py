# src/build_raw_corpus.py
import re
from pathlib import Path
from datasets import load_dataset

# Articles to keep: période d'essai and congés payés
WANTED = {f"L1221-{n}" for n in range(19, 27)} | {f"L3141-{n}" for n in range(1, 13)}

ds = load_dataset("louisbrulenaudet/code-travail", split="train")
print(ds)  # check the column names first

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

found = set()
for row in ds:
    # Full article number, including sub-articles like L3141-5-1
    match = re.search(r"L\d{4}-\d+(?:-\d+)*", row["ref"])
    if match and match.group() in WANTED:
        article_id = match.group()
        if article_id in found:
            raise ValueError(f"Duplicate row for {article_id}: {row['ref']}")
        text = f"Article {article_id} - Code du travail\n{row['texte']}"
        (out_dir / f"{article_id}.txt").write_text(text, encoding="utf-8")
        found.add(article_id)

print(f"Saved {len(found)} articles")
print("Missing:", sorted(WANTED - found))