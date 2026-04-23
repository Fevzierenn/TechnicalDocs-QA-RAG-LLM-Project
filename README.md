#  Java Spring Technical Documentation QA System
## Retrieval-Augmented Generation (RAG) with Deep Learning
 

##  Project Overview

Large Language Models (LLMs) show strong natural language generation abilities, but they suffer from critical limitations when applied to **technical and domain-specific documentation**, such as Java and Spring Framework.

These limitations include:
- Knowledge cutoff
- Hallucination (fabricated or ungrounded answers)
- Lack of source traceability
- Poor faithfulness to enterprise-level documentation

This project designs, implements, and **scientifically evaluates** a **Retrieval-Augmented Generation (RAG)** based Question Answering (QA) system over **Java & Spring technical documentation**, primarily sourced from **Baeldung**.

The system combines:
- Structure-aware document chunking
- Semantic vector retrieval
- Local and API-based Large Language Models
- Multi-stage quantitative and qualitative evaluation pipelines

---

## Problem Definition

### Why RAG Is Necessary

Standard LLMs rely solely on **parametric memory** learned during training. This leads to three major problems:

### 1. Hallucination
LLMs generate confident but incorrect answers, which is unacceptable in syntax-sensitive frameworks like Spring.

### 2. Knowledge Cutoff
Fine-tuning models for each framework update is costly and impractical.

### 3. Lack of Grounding
LLMs cannot indicate which document their answers are based on, making verification impossible.

---

##  Solution Approach

This project implements a **domain-specific RAG pipeline**:

1. A verified **Knowledge Base** built from Java & Spring articles  
2. Semantic retrieval using **vector embeddings and ChromaDB**  
3. Context-aware answer generation using:
   - Local LLMs (Qwen, Mistral, Gemma, Llama)
   - API-based LLM (Llama‑3.3‑70B via Groq)

Performance is measured using:
- Exact Match (EM)
- F1 Score
- Recall@K
- Mean Reciprocal Rank (MRR)
- Latency
- Hallucination Rate

---

##  Knowledge Base Construction

### Data Source
- 120 technical articles
- Scraped using **Python + Selenium** (dynamic DOM rendering)

### Preprocessing Pipeline
- Removal of ads, navigation bars, sidebars, footers, comments
- HTML tag stripping
- Code block preservation
- Encoding and whitespace normalization

### Dataset Statistics

| Metric | Value |
|------|------|
| Documents | 120 |
| Pages (Equivalent) | 300+ |
| Content Type | Text + Java Code Snippets |

---

##  Chunking Strategies

### Fixed-Length Chunking (Control Group)
- Chunk Size: 1000 characters
- Overlap: 200 characters
- Tool: RecursiveCharacterTextSplitter

### Smart / Section-Based Chunking ✅
- Tool: MarkdownHeaderTextSplitter
- Preserves document hierarchy and code-context integrity
- Selected as final strategy

---

## 🔎 Embedding Models Evaluation

### Tested Models
- sentence-transformers/all-MiniLM-L6-v2
- BAAI/bge-small-en-v1.5

### Final Selection
 **Smart Chunking + MiniLM-L6-v2**

| Configuration | Recall@1 | Recall@3 | Encoding Time |
|-------------|---------|---------|--------------|
| Fixed + MiniLM | 80% | 86% | 40.54s |
| Smart + MiniLM | **88%** | **94%** | **39.53s** |
| Smart + BGE | 80% | 90% | 87.86s |

---

##  Vector Database Configuration

- **Vector DB:** ChromaDB (Persistent)
- **Index Type:** HNSW
- **Similarity Metric:** Cosine Similarity (Inner Product optimized)
- **Embedding Dimension:** 384
- **Top-K:** 3

HNSW was selected due to its logarithmic query complexity and low-latency performance.

---

##  Retriever-Only Evaluation

Retriever performance was evaluated independently before introducing LLMs.

| Metric | Value |
|------|------|
| Recall@1 | 90% |
| Recall@3 | 96% |
| MRR@3 | 0.923 |

This confirms that most downstream errors are **LLM-related**, not retriever-related.

---

##  LLM-Only (No-RAG) Performance

| Metric | Value |
|------|------|
| F1 Score | 0.157 |
| Human Relevance | 3.96 / 5 |
| Faithfulness | 3.49 / 5 |
| Hallucination Rate | **58%** |

LLMs without RAG are unreliable for technical QA.

---

##  Full RAG System Performance

| Metric | No-RAG | RAG | Improvement |
|------|------|------|------------|
| F1 Score | 0.157 | **0.395** | +151% |
| Faithfulness | 3.49 | **4.85** | ↑ |
| Hallucination Rate | 58% | **5%** | −90% |

RAG transforms LLMs from generative chatbots into grounded knowledge systems.

---

##  Best Configuration (Golden Setup)

- **Embedding:** all-MiniLM-L6-v2  
- **Chunking:** Smart / Section-Based  
- **Retriever:** ChromaDB + HNSW  
- **Top-K:** 3  
- **LLM:** Qwen-8B (Quantized)

**Results:**
- F1 Score: **0.427**
- Hallucination: **0%**
- Tradeoff: High latency (~45s)

---

##  Hallucination & Failure Analysis

The project includes detailed failure case studies:
- False Refusal
- Lazy Retrieval
- Parametric Memory Leakage
- Contextual Overflow
- Ungrounded Correct Answers

Each failure is analyzed with root-cause explanations.

---

##  Future Work

- Self-Correction (Judge LLM)
- GraphRAG (Class → Method → Annotation)
- Cross-Encoder Re-ranking
- Multi-modal RAG (UML diagrams)
- Multi-hop reasoning

---

##  Final Conclusion

This project demonstrates that:

- A well-designed RAG system is more important than model size
- Chunking strategy has a greater impact than embedding model choice
- Local 8B models with RAG can rival 70B API models
- RAG reduces hallucination by **over 90%**

---

##  License

MIT License
