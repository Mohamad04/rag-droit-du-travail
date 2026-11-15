import dotenv
from search import search, chunk_texts


def build_context(results):
    context = ""
    for chunk_id, _ in results:
        article_id = chunk_id.split("#")[0]
        text = chunk_texts[chunk_id]
        context += f"[{article_id}]\n{text}\n\n"
    return context 

print(build_context([("L1221-23#0", 0.9), ("L1221-21#0", 0.8)]))
