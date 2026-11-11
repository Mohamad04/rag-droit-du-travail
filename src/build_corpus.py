import json
from pathlib import Path

TOPIC = {
    "L1221": "periode_essai",
    "L3141": "conges_payes",
}

raw_dir = Path("data/raw")
output_path = Path("data/corpus.jsonl")

with open(output_path, "w", encoding="utf-8") as out:
    for path in sorted(raw_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        text = raw.split("\n", 1)[1].strip()   

        article_id = path.stem                  
        prefix = article_id.split("-")[0]       
        if prefix not in TOPIC:
            raise ValueError(f"No topic defined for article {article_id}")

        record = {
            "id": article_id,
            "code": "Code du travail",
            "topic": TOPIC[prefix],
            "text": text,
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote corpus to {output_path}")