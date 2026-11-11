from pathlib import Path
import json
import statistics

data_path = Path("data/corpus.jsonl")

n_words = 0
lengths = {}


with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                record = json.loads(line)
                n_words = len(record["text"].split())
                lengths[record["id"]] = n_words
            except json.JSONDecodeError as e:
                print(f"Error parsing line:{e}")
    for article_id, n_words in sorted(lengths.items(), key=lambda x: x[1], reverse=True):
        print(article_id, n_words)
print(f"Median: {statistics.median(lengths.values())} words")
