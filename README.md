# RAG on French Labor Law (droit du travail)

A retrieval-augmented generation (RAG) system that answers questions about French labor law, **built from scratch without frameworks** and **evaluated with retrieval metrics**.

> Status: work in progress.

## Goal

Understand every step of a RAG pipeline by implementing it myself, then measure how each design choice (chunking, embedding model, search method) affects retrieval quality. A second version built with LangChain will be compared against it on the same evaluation set.

## Scope

- **Corpus:** articles of the *Code du travail* and the *Code de l'éducation* on three topics:
  internships (*stages*), paid leave (*congés payés*) and trial periods (*période d'essai*).
  Source: Légifrance. Each article is stored as one text file named after its number (e.g. `L1221-19.txt`).
- **Evaluation set:** questions written from the official explanations on service-public.fr,
  each with its expected answer and the article(s) that should be retrieved.

## Plan

1. Corpus collection
2. Chunking (size and overlap studied as parameters)
3. Embeddings with `sentence-transformers`
4. Vector search: NumPy cosine similarity, then FAISS
5. Generation with a local LLM (Ollama), with source citations
6. Evaluation: hit rate and MRR on the evaluation set
7. Comparison with a LangChain implementation

## Project structure

```
data/raw/   legal texts, one file per article
src/        pipeline code
eval/       evaluation questions and scripts
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Disclaimer

This is a learning project. Its answers are not legal advice.