import dotenv
from search import search, chunk_texts
from openai import OpenAI

REFUSAL = "Le corpus ne contient pas la réponse à cette question."

SYSTEM_PROMPT = f"""Tu es un assistant qui répond aux questions sur le Code du travail français.
Voici les règles que tu dois suivre :
1. Réponds toujours dans la langue de la question : si la question est en anglais, réponds en anglais. Seule la phrase de refus reste en français.
2. Réponds uniquement à partir des articles fournis, jamais à partir de tes connaissances.
3. Cite la source de chaque affirmation avec l'identifiant exact entre crochets, par exemple [L1221-23].
4. Si les articles ne répondent qu'en partie, réponds avec ce qu'ils contiennent et précise ce qui n'y figure pas.
   Ne refuse que si aucun article ne traite du sujet de la question. Dans ce cas, réponds exactement par cette phrase, et rien d'autre : {REFUSAL}
5. Ne cite jamais un article qui ne figure pas dans les articles fournis.
6. Réponds de manière claire et brève, pour quelqu'un qui n'est pas juriste.
"""

dotenv.load_dotenv()
client = OpenAI()

def build_context(results):
    context = ""
    for chunk_id, _ in results:
        article_id = chunk_id.split("#")[0]
        text = chunk_texts[chunk_id]
        context += f"[{article_id}]\n{text}\n\n"
    return context 

def generate(question):
    best_chunks = search(question=question,use_reranker=True)

    context = build_context(best_chunks)
    # 3. user_message: articles + question
    user_message = f"Articles :\n\n{context}\nQuestion : {question}" 
    # 4. messages: system + user
    messages = [
            {
                "role" : "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    
    # 5. OpenAI call
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages
    )
    # 6. return the model's answer (a string)
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "I'm a manager, and my employee is asking me for a day off. Am I allowed to refuse his request?"
    print(generate(question))
