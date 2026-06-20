# RAG Pipeline for Narrative Question Answering

A Retrieval-Augmented Generation (RAG) pipeline built over a public-domain novel
(from Project Gutenberg) and the [`deepmind/narrativeqa`](https://huggingface.co/datasets/deepmind/narrativeqa)
benchmark. The project compares **two chunking strategies** end-to-end and
evaluates their impact on answer quality using **BLEU** and **ROUGE-L**.

 Demonstrates a full RAG stack:
> data preprocessing → chunking → embeddings → vector search → reranking →
> LLM generation → evaluation.

## Pipeline Overview

1. **Data Preprocessing** — Downloads the raw book text, strips Project Gutenberg
   boilerplate (license, headers, footers), and filters NarrativeQA down to the
   116 question–answer pairs relevant to the book. Encapsulated in a
   `DataPreprocessor` class.
2. **Chunking (two strategies, compared)**
   - *Fixed-size character chunking* — sliding window with overlap.
   - *Sentence-based chunking* — NLTK sentence tokenization to preserve semantic
     boundaries.
3. **Embeddings** — `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors).
4. **Vector Database** — [Milvus Lite](https://milvus.io/) with an idempotent,
   batch-encoded ingestion pipeline (drops + recreates collections for
   reproducible runs).
5. **Retrieval + Reranking** — Top-K semantic search in Milvus, followed by a
   `CrossEncoder` reranker that scores each (query, passage) pair.
6. **Generation** — `Qwen/Qwen2.5-0.5B-Instruct` produces the final answer from
   the retrieved context (prompt-engineered).
7. **Evaluation** — Average **BLEU** (with smoothing) and **ROUGE-L F1** across
   all questions, for both chunking strategies.

## Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `sentence-transformers` ·
`Milvus (pymilvus)` · `NLTK` · `datasets` · `rouge_score` · `matplotlib`

## Results

The two chunking strategies are compared on the same 116-question set:

| Strategy             | BLEU            | ROUGE-L F1      |
|----------------------|-----------------|-----------------|
| Fixed-size chunking  | 0.0115 | 0.0585|
| Sentence-based       | 0.0099| 0.0577 |
On this small-model setup, fixed-size chunking performed marginally better than sentence-based across both metrics, suggesting chunk boundary strategy had limited impact at this scale.

## How to Run

```bash
# (recommended) create a virtual environment first
pip install -r requirements.txt
jupyter notebook rag_pipeline.ipynb
```

Run the cells top to bottom. A GPU speeds up embedding and generation but is not
required.

## Repository Structure

```
.
├── rag_pipeline.ipynb   # main notebook (full pipeline)
├── report.pdf           # written project report
├── requirements.txt
└── README.md
```

## Key Takeaways

- End-to-end RAG implementation, not just a single component.
- Controlled experiment: isolates chunking strategy as the variable and measures
  its effect with standard NLP metrics.
- Modular, class-based code (`DataPreprocessor`, `TextChunker`) and a
  reproducible vector-DB ingestion pipeline.
