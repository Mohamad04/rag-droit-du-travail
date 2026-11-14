import numpy as np
import json
from build_chunks import load_corpus
from build_index import MODEL_NAME,CHUNK_IDS_PATH,EMBEDDINGS_PATH,CHUNKS_PATH
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_index(path):
    return np.load(path)

def load_chunk_ids(path):
    with open(path,"r",encoding="utf-8") as f:
        chunk_ids = json.load(f)
    return chunk_ids


model = SentenceTransformer(MODEL_NAME)
embeddings = load_index(EMBEDDINGS_PATH)
chunk_ids = load_chunk_ids(CHUNK_IDS_PATH)
chunk_texts = {c["chunk_id"]: c["text"] for c in load_corpus(CHUNKS_PATH)}

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
reranker = CrossEncoder(RERANK_MODEL)

def rerank_candidates(question, candidate_ids, k):
    pairs = [(question, chunk_texts[chunk_id]) for chunk_id in candidate_ids]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)

    return [(chunk_id, float(score)) for chunk_id, score in ranked[:k]]


def search(question, k=5, use_reranker=False, n_candidates=20):
    query_vector = model.encode(f"query: {question}", normalize_embeddings=True)
    scores = embeddings @ query_vector
    top_n = np.argsort(scores)[::-1][:n_candidates]
    results = [(chunk_ids[i], float(scores[i])) for i in top_n]

    if not use_reranker:
        return results[:k]

    candidate_ids = [chunk_id for chunk_id, _ in results]
    return rerank_candidates(question, candidate_ids, k)

def main():
    question = input("Votre question : ")
    for rank, (chunk_id, score) in enumerate(search(question), start=1):
        print(f"{rank}. {chunk_id}  (score: {score:.3f})")

if __name__ == "__main__":
    main()