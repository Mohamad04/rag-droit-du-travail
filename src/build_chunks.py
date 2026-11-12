import argparse
import json
from pathlib import Path

CORPUS_PATH = Path("data/corpus.jsonl")
OUT_PATH = Path("data/chunks,jsonl")

def load_corpus(path):
    """Read corpus.jsonl and return a list of article dicts."""
    # TODO: read the file line by line, json.loads each line, return the list
    articles = []
    with open(path,"r",encoding="utf-8") as corpus:
        for line in corpus:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles



def chunk_v1(article):
    """One chunk per article. Returns a list of chunk dicts."""
    # TODO: return a list containing ONE chunk dict:
    # chunk_id, article_id, topic, text
    record_chunk = {
        "chunk_id": f"{article['id']}#0",
        "article_id": article["id"],
        "topic": article["topic"],
        "text": f"Article {article['id']}: {article["text"]}"
    }
    return [record_chunk]


STRATEGIES = {
    "v1": chunk_v1,
    # "v2": chunk_v2,   # added later
}


def write_chunks(chunks, path):
    """Write the chunks to a JSONL file."""
    # TODO: open once, json.dumps each chunk with ensure_ascii=False
    with open(path,"w", encoding="utf-8") as out:
        for chunk in chunks:
            out.write(json.dumps(chunk,ensure_ascii=False)+ "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGIES.keys(), default="v1")
    args = parser.parse_args()

    articles = load_corpus(CORPUS_PATH)
    chunk_fn = STRATEGIES[args.strategy]

    chunks = []
    for article in articles:
        chunks.extend(chunk_fn(article))

    output_path = Path(f"data/chunks_{args.strategy}.jsonl")
    write_chunks(chunks, output_path)
    print(f"Wrote {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()