from pathlib import Path

from build_chunks import load_corpus
from search import search

QUESTIONS_PATH = Path("eval/questions.jsonl")
CHUNKS_PATH = Path("data/chunks_v1.jsonl")
K = 5


def first_hit_rank(retrieved_articles, expected_articles):
    """Rank of the first expected article, or None."""
    for rank, article_id in enumerate(retrieved_articles, start=1):
        if article_id in expected_articles:
            return rank
    return None


def all_hit_ranks(retrieved_articles, expected_articles):
    """Ranks of all expected articles found in the results."""
    return [rank for rank, article_id in enumerate(retrieved_articles, start=1)
            if article_id in expected_articles]
            


        


def main():
    chunk_to_article = {c["chunk_id"]: c["article_id"] for c in load_corpus(CHUNKS_PATH)}
    questions = [q for q in load_corpus(QUESTIONS_PATH) if q["expected_articles"]]

    ranks = []
    for q in questions:
        results = search(q["question"], k=K)
        retrieved_articles = [chunk_to_article[chunk_id] for chunk_id, _ in results]
        rank = first_hit_rank(retrieved_articles, q["expected_articles"])
        ranks.append(rank)
        if rank != 1:
            print(f"\n{q['id']}: {q['question']}")
            print(f"  expected {q['expected_articles']}, got rank {rank}")
            for i, (chunk_id, score) in enumerate(results, start=1):
                print(f"  {i}. {chunk_to_article[chunk_id]}  ({score:.3f})")

    # TODO: compute and print
    #   hit rate @1, @3, @5: fraction of ranks that are not None and <= k
    #   MRR: average of 1/rank, counting 0 when rank is None
    n = len(ranks)
    for k in [1, 3, 5]:
        hit = sum(1 for r in ranks if r is not None and r <= k) / n
        print(f"Hit rate @{k}: {hit:.3f}")
    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"MRR average: {mrr:.3f}")


if __name__ == "__main__":
    main()