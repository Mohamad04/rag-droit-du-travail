# RAG on French Labor Law (droit du travail)

A retrieval-augmented generation (RAG) system that answers questions about French labor law, **built from scratch without frameworks** and **evaluated with retrieval metrics**.

> Status: work in progress. Retrieval and evaluation work; generation is not implemented yet.

## Goal

Understand every step of a RAG pipeline by implementing it myself, then measure how each design choice (chunking, embedding model, search method) affects retrieval quality. A second version built with LangChain will be compared against it on the same evaluation set.

## Scope

- **Corpus:** 20 articles of the *Code du travail*, from Légifrance (via the `louisbrulenaudet/code-travail` dataset), on two topics:
  - trial period (*période d'essai*): `L1221-19` to `L1221-26`
  - paid leave (*congés payés*): `L3141-1` to `L3141-12`

  Each article is stored as one text file named after its number (e.g. `L1221-19.txt`).
- **Evaluation set:** 15 questions in `eval/questions.jsonl` (French and English), written from the official explanations on service-public.fr. Each has an expected answer and the article(s) that should be retrieved. 4 questions have no answer in the corpus (`expected_articles` is empty) and are skipped by the retrieval metrics, so **11 questions are scored**.

## Plan

1. Corpus collection ✅
2. Chunking (size and overlap studied as parameters): v1 done, other strategies to come
3. Embeddings with `sentence-transformers` ✅ (`intfloat/multilingual-e5-small`)
4. Vector search: NumPy cosine similarity ✅, then FAISS
5. Generation with a local LLM (Ollama), with source citations
6. Evaluation: hit rate and MRR on the evaluation set ✅ (baseline recorded below)
7. Comparison with a LangChain implementation

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| Download articles | `src/build_raw_corpus.py` | `data/raw/*.txt` |
| Build corpus | `src/build_corpus.py` | `data/corpus.jsonl` |
| Chunk | `src/build_chunks.py --strategy v1` | `data/chunks_v1.jsonl` |
| Embed and index | `src/build_index.py` | `data/index/` (not versioned) |
| Search | `src/search.py` | top-k chunks for a question |
| Evaluate | `src/evaluate.py` | hit rate @1/3/5 and MRR |

Current configuration:

- **Chunking v1:** one chunk per article, prefixed with `Article <id>:`.
- **Embedding model:** `intfloat/multilingual-e5-small`, with the `passage: ` prefix for documents and `query: ` for questions, normalized embeddings.
- **Search:** cosine similarity (dot product of normalized vectors) over a NumPy matrix, top 5.

## Results

### V1 (baseline)

Chunking v1 (one chunk per article) + `multilingual-e5-small` + NumPy cosine search, on the 11 scored questions (k = 5).

| Metric | Value |
|--------|-------|
| Hit rate @1 | 0.545 |
| Hit rate @3 | 0.727 |
| Hit rate @5 | 0.818 |
| MRR | 0.655 |

Questions where the expected article was not ranked first, with the top 5 retrieved (cosine score in parentheses):

| Question | Expected | Rank found | Top 5 retrieved |
|----------|----------|-----------|-----------------|
| q05: employer imposing a trial period not written in the contract | `L1221-23` | 2 | L1221-19 (0.878), **L1221-23 (0.875)**, L1221-26 (0.861), L1221-21 (0.860), L1221-22 (0.858) |
| q06: "Chez KFC, la période d'essai est-elle obligatoire ?" | `L1221-23` | 5 | L1221-21 (0.849), L1221-26 (0.847), L1221-22 (0.846), L1221-19 (0.841), **L1221-23 (0.839)** |
| q07: "la période d'essai, c'est la période pendant laquelle le salarié essaie l'entreprise ?" | `L1221-20` | none | L1221-26 (0.891), L1221-21 (0.879), L1221-19 (0.875), L1221-25 (0.865), L1221-22 (0.864) |
| q08: "la période d'essai, c'est un jour férié ?" | `L1221-20` | none | L1221-26 (0.855), L1221-22 (0.853), L1221-21 (0.853), L1221-25 (0.847), L1221-19 (0.846) |
| q09: manager refusing a day off (English) | `L3141-1`, `L3141-12` | 2 | L3141-6 (0.817), **L3141-12 (0.811)**, L3141-3 (0.804), L3141-7 (0.800), L3141-2 (0.799) |

Observations (the known problems of V1):

- **Scores barely discriminate.** Within a query, the top 5 scores span only 0.01 to 0.04 (e.g. 0.859 to 0.839 for q06), so the ranking among articles of the same topic is close to arbitrary.
- **Topic is found, article is not.** For q05 to q08 every retrieved article is a `L1221-*` trial-period article, and for q09 all are `L3141-*` paid-leave articles. The model recognizes the topic but does not pick the right article.
- **`L1221-20` is never retrieved** for q07 and q08, which are loosely phrased definition questions.
- Failures concentrate on vague or informal questions (q06, q07, q08) and the one English question (q09).
- With only 20 articles, a top 5 already covers 25% of the corpus, so hit rate @5 is optimistic. Hit rate @1 and MRR are the more informative numbers.
- With 11 scored questions, one question moves hit rate by about 9 points; treat differences between strategies cautiously.

## Project structure

```
data/raw/            legal texts, one file per article
data/corpus.jsonl    one JSON record per article
data/chunks_v1.jsonl chunks produced by strategy v1
data/index/          embeddings and chunk ids (generated, not versioned)
src/                 pipeline code
eval/                evaluation questions
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`data/raw/`, `data/corpus.jsonl` and `data/chunks_v1.jsonl` are versioned. Only `data/index/` must be generated. To run the whole pipeline from scratch:

```bash
uv run python src/build_raw_corpus.py   # downloads the articles into data/raw/
uv run python src/build_corpus.py       # builds data/corpus.jsonl
uv run python src/build_chunks.py       # builds data/chunks_v1.jsonl
uv run python src/build_index.py        # embeds chunks into data/index/
uv run python src/evaluate.py           # prints hit rate and MRR
uv run python src/search.py             # interactive search
```

## Disclaimer

This is a learning project. Its answers are not legal advice.
