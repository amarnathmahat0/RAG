# NexusRAG — Production-Grade Multi-Agent RAG System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LLM](https://img.shields.io/badge/LLM-Ollama-orange)
![Models](https://img.shields.io/badge/Models-gemma3%20%7C%20phi3%20%7C%20qwen-purple)
![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent-green)
![RAG](https://img.shields.io/badge/RAG-Hybrid%20%2B%20Graph-blueviolet)
![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-red)
![GraphDB](https://img.shields.io/badge/GraphDB-Neo4j-brightgreen)
![Evaluation](https://img.shields.io/badge/Evaluation-LLM%20Judge-yellow)
![API](https://img.shields.io/badge/API-FastAPI-teal)
![Cost](https://img.shields.io/badge/API%20Cost-₹0-success)
![License](https://img.shields.io/badge/License-MIT-yellow)
## 🚀 Overview

NexusRAG is a production-grade, multi-agent RAG system designed to deliver accurate, fast, and explainable responses — without relying on any paid APIs.

It combines:

* Adaptive query routing
* Hybrid search (dense + keyword)
* Graph-based reasoning
* LLM-based evaluation (fully local)

---

## ✨ Key Features

* ⚡ **3-Tier Adaptive Routing**

  * TIER 1: Fast hybrid retrieval (<500ms)
  * TIER 2: Multi-step reasoning (<1.5s)
  * TIER 3: GraphRAG (Neo4j traversal) (<3s)

* 🔍 **Hybrid Retrieval Pipeline**

  * Dense embeddings + BM25
  * Reciprocal Rank Fusion (RRF)
  * FlashRank reranking

* 🧠 **Multi-Agent Architecture**

  * RouterAgent
  * RetrieverAgent
  * ResponseBuilder
  * CriticAgent (LLM judge)
  * ComplianceAgent
  * FormatterAgent

* 🧪 **LLM-as-Judge Evaluation**

  * Faithfulness
  * Relevancy
  * Completeness

* 🔐 **Compliance Layer**

  * PII detection
  * Prompt injection protection

* 📊 **Production Observability**

  * Prometheus metrics
  * JSONL tracing
  * Request-level logging

* 💰 **Zero API Cost**

  * 100% local using Ollama

---

## 🧱 Architecture

### High-Level Flow

```
User Query
   ↓
Compliance Check (PII / Injection)
   ↓
Adaptive Router (rule-based + LLM scoring)
   ↓
Tier Selection:
   ├── TIER 1 → Fast Hybrid Search
   ├── TIER 2 → Query Decomposition
   └── TIER 3 → GraphRAG (Neo4j)
   ↓
Retriever (Dense + BM25 → RRF → Rerank)
   ↓
Response Generation (LLM)
   ↓
Critic Agent (Faithfulness Check)
   ↓
Post Compliance Check
   ↓
Formatter (citations + confidence)
   ↓
Final Response
```

---

## ⚙️ Tech Stack

| Layer       | Technology         |
| ----------- | ------------------ |
| Backend     | FastAPI            |
| LLM Runtime | Ollama             |
| Vector DB   | ChromaDB           |
| Graph DB    | Neo4j              |
| Retrieval   | BM25 + Dense + RRF |
| Monitoring  | Prometheus         |
| Evaluation  | Custom + LLM judge |
| CI/CD       | GitHub Actions     |

---

## 🧠 Model Roles

| Model      | Size  | Role                      |
| ---------- | ----- | ------------------------- |
| all-minilm | 45MB  | Embeddings (primary)      |
| gemma3:1b  | 815MB | Routing + query transform |
| phi3:mini  | 2.2GB | Answer generation         |
| qwen:1.8b  | 1.1GB | Evaluation / critic       |

> Only one model is loaded at a time → keeps RAM usage under ~4GB

---

## ⚡ Quick Start

### 1. Install Ollama

```bash
brew install ollama
ollama serve
```

### 2. Pull Models

```bash
ollama pull all-minilm:latest
ollama pull gemma3:1b
ollama pull qwen:1.8b
ollama pull phi3:mini
```

### 3. Start Infrastructure

```bash
make up
```

### 4. Install Dependencies

```bash
make install
make spacy-model
```

### 5. Ingest Documents

```bash
cp your_file.pdf data/raw/
make ingest
```

### 6. Run API

```bash
make run
```

API Docs: http://localhost:8000/docs

---

## 📡 API Usage

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "What is machine learning?",
        "stream": false
      }'
```

### Example Response

```json
{
  "request_id": "abc123",
  "answer": "Machine learning is a subset of AI...",
  "tier": "TIER_1",
  "confidence": 0.84,
  "latency_ms": 312,
  "faithfulness_score": 0.92
}
```

---

## 🧪 Evaluation System

### Deterministic Metrics

* Context Precision
* Context Recall
* MRR (Mean Reciprocal Rank)
* NDCG@5
* Citation Accuracy
* Latency (p50, p95, p99)

### LLM-Based Evaluation

* Faithfulness (fact-check vs context)
* Relevancy (question alignment)
* Completeness (coverage)

---

## 📊 CI/CD Gates

Every PR must pass:

* ✅ Lint (ruff)
* ✅ Type check (mypy)
* ✅ Unit tests (pytest)
* ✅ Evaluation thresholds:

  * Precision ≥ 0.6
  * Recall ≥ 0.6
  * Latency p95 < 5s

```bash
make ci-check
```

---

## 📈 Monitoring & Observability

* Prometheus metrics endpoint (`/metrics`)
* Query latency tracking
* Hallucination rate monitoring
* RAM usage tracking

---

## 📁 Project Structure

```
nexus-rag/
├── src/
│   ├── agents/        # Multi-agent system
│   ├── retrieval/     # Hybrid search logic
│   ├── generation/    # Response building
│   ├── evaluation/    # Metrics + LLM judge
│   ├── api/           # FastAPI app
│   └── utils/         # Config + ModelManager
├── data/
├── tests/
├── infra/
└── Makefile
```

---

## 🛠️ Makefile Commands

```bash
make up
make run
make test
make eval
make ingest
make ci-check
```

---

## 🎯 Design Goals

* Low latency with adaptive routing
* High accuracy with hybrid retrieval
* Zero operational cost
* Fully local execution
* Production-ready architecture

---

## 📜 License

MIT License

---

## 💡 Author

GitHub: https://github.com/amarnathmahat0

---

## ⭐ If you find this useful, consider giving a star!
