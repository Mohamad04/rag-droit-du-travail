import dotenv
from search import search, chunk_texts
from openai import OpenAI
from langdetect import detect, DetectorFactory, LangDetectException
import logging

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

REFUSAL = "Le corpus ne contient pas la réponse à cette question."

SYSTEM_PROMPT = f"""Tu es un assistant qui répond aux questions sur le Code du travail français.
Voici les règles que tu dois suivre :
1. Réponds uniquement à partir des articles fournis, jamais à partir de tes connaissances.
2. Cite la source de chaque affirmation avec l'identifiant exact entre crochets, par exemple [L1221-23].
3. Si les articles ne répondent qu'en partie, réponds avec ce qu'ils contiennent et précise ce qui n'y figure pas.
   Ne refuse que si aucun article ne traite du sujet de la question. Dans ce cas, réponds exactement par cette phrase, et rien d'autre : {REFUSAL}
4. Ne cite jamais un article qui ne figure pas dans les articles fournis.
5. Réponds de manière claire et brève, pour quelqu'un qui n'est pas juriste.
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

def answer_instruction(question):
    try:
        lang_code = detect(question)
        logger.debug("Detected language: %s", lang_code)
    except LangDetectException as e:
        logger.warning("Language detection failed (%s); defaulting to French", e)
        lang_code = "fr"
    return f"Reply in the language this code belongs to: {lang_code}, unless the user asks you answer in another language"
    
def generate(question):
    best_chunks = search(question=question,use_reranker=True)

    context = build_context(best_chunks)
    # 3. user_message: articles + question
    user_message = f"Articles :\n\n{context}\nQuestion : {question}\n\n{answer_instruction(question)}"
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
    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    question = "I'm a manager, and my employee is asking me for a day off. Am I allowed to refuse his request?"
    print(generate(question))