import numpy as np
from pathlib import Path
import json
from build_index import MODEL_NAME,CHUNK_IDS_PATH,EMBEDDINGS_PATH
from sentence_transformers import SentenceTransformer



def load_index(path):
    return np.load(path)

def load_chunk_ids(path):
    with open(path,"r",encoding="utf-8") as f:
        chunk_ids = json.load(f)
    return chunk_ids


model = SentenceTransformer(MODEL_NAME)
embeddings = load_index(EMBEDDINGS_PATH)
chunk_ids = load_chunk_ids(CHUNK_IDS_PATH)



def search(question, k =5):
    query_vector = model.encode(f"query: {question}",normalize_embeddings=True)
    scores = embeddings @ query_vector
    top_k = np.argsort(scores)[::-1][:k]
    return [(chunk_ids[i],float(scores[i])) for i in top_k]

def main():
    question = input("Votre question : ")
    for rank, (chunk_id, score) in enumerate(search(question), start=1):
        print(f"{rank}. {chunk_id}  (score: {score:.3f})")

if __name__ == "__main__":
    main()