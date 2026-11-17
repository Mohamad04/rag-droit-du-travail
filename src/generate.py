import logging

import dotenv
import pycountry
from langdetect import DetectorFactory, LangDetectException, detect
from openai import OpenAI

from search import chunk_texts, search

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

MODEL = "gpt-5.4-mini"

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


def language_name(lang_code):
    language = pycountry.languages.get(alpha_2=lang_code.split("-")[0])
    return language.name if language else lang_code


def detect_language(question):
    """Return the ISO code of the question's language, or "fr" if detection fails."""
    try:
        lang_code = detect(question)
    except LangDetectException as e:
        logger.warning("Language detection failed (%s); defaulting to French", e)
        return "fr"
    logger.debug("Detected language: %s (%s)", lang_code, language_name(lang_code))
    return lang_code


def build_user_message(question, context, lang_code):
    # The question comes LAST: the model gives the most weight to what it reads last,
    # so an explicit language request in the question must come after our default.
    return (
        f"Articles :\n\n{context}"
        f"Langue de réponse par défaut : {language_name(lang_code)}. "
        f"Si la question demande explicitement une autre langue, réponds dans cette langue.\n\n"
        f"Question : {question}"
    )


def generate(question):
    logger.info("Question: %s", question)

    # 1. Retrieval
    results = search(question=question, use_reranker=True)
    logger.debug("Retrieved %d chunks:", len(results))
    for rank, (chunk_id, score) in enumerate(results, start=1):
        logger.debug("  %d. %s (score %.3f)", rank, chunk_id, score)

    # 2. Context, language and user message
    context = build_context(results)
    lang_code = detect_language(question)
    user_message = build_user_message(question, context, lang_code)
    logger.debug("User message sent to the model:\n%s", user_message)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # 3. LLM call
    response = client.chat.completions.create(model=MODEL, messages=messages)
    choice = response.choices[0]
    logger.debug("Finish reason: %s", choice.finish_reason)
    if response.usage:
        logger.debug(
            "Tokens: prompt=%d, completion=%d",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    answer = choice.message.content or ""
    logger.info("Refusal: %s", answer.strip() == REFUSAL)
    return answer


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # DEBUG applies to every library too: keep only our own debug messages.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "urllib3",
                  "sentence_transformers", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    test_questions = [
        "I'm a manager, and my employee is asking me for a day off. Am I allowed to refuse his request? Please answer in arabic",
        "I'm a manager, and my employee is asking me for a day off. Am I allowed to refuse his request?",
        "Chez KFC, la période d'essai est-elle obligatoire ?",
    ]

    # The VS Code terminal can't display Arabic properly: also write the answers to a file.
    with open("answers.md", "w", encoding="utf-8") as f:
        for question in test_questions:
            answer = generate(question)
            print(f"\n{answer}\n")
            f.write(f"## {question}\n\n{answer}\n\n")