# 🚀 Quick Start - Verify Your Updates

This guide helps you verify all new features are working correctly.

---

## ✅ Pre-flight Checklist

### 1. Start the Services

```bash
# Terminal 1: Start databases
make up

# Terminal 2: Start API
make run

# Terminal 3: Start Streamlit UI
streamlit run app.py
```

### 2. Verify Health Check

```bash
curl http://localhost:8000/health

# Expected response with all green:
{
  "status": "ok",
  "components": {
    "api": {"status": "ok"},
    "chromadb": {"status": "ok"},
    "neo4j": {"status": "ok"},
    "models": {"status": "ok", "free_ram_mb": 12000, ...}
  }
}
```

---

## 📝 Test File Upload

### Method 1: Streamlit UI (Easiest)

1. Open http://localhost:8501
2. In sidebar under "Upload Documents":
   - Drag & drop files or click "Choose files"
   - Click "⬆ Ingest Files"
   - Wait for success messages

### Method 2: API Endpoint

```bash
# Upload a single file
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "files=@test_document.pdf"

# Upload multiple files
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "files=@doc3.txt"

# Expected response:
# {
#   "accepted": 3,
#   "sources": ["/path/to/doc1.pdf", ...],
#   "message": "Upload complete..."
# }
```

---

## 📚 Test Document Selection

### Verify Documents Imported

```bash
curl http://localhost:8000/documents

# Expected:
# {
#   "documents": [
#     {"name": "doc1.pdf", "source": "...", "chunks": 128},
#     {"name": "doc2.docx", "source": "...", "chunks": 45}
#   ]
# }
```

### Query All Documents (Default)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "stream": false
  }'
```

### Query Specific Document

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "documents": ["doc1.pdf"],
    "stream": false
  }'
```

### Query Multiple Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare these documents",
    "documents": ["doc1.pdf", "doc2.docx"],
    "stream": false
  }'
```

---

## ⚡ Test Latency Optimization

### First Query (No Cache)

```bash
time curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "stream": false}' \
  -s | jq '.latency_ms'

# Expected: 400-1500ms
# Response should include latency_ms field
```

### Same Query Again (With Cache)

```bash
time curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "stream": false}' \
  -s | jq '.latency_ms'

# Expected: <50ms
# Notice the dramatic speed improvement!
```

### Check Cache Directory

```bash
ls -lh data/cache/

# Should see .json files for cached queries/embeddings
# Each file is a hash-based cache entry
```

---

## 🔧 Verify Configuration

### Models Configured Correctly

```bash
python3 << 'EOF'
from src.utils.config import get_settings

settings = get_settings()
print("=== Model Configuration ===")
print(f"Embedding Model: {settings.embed_model_primary}")
print(f"LLM Router: {settings.llm_router}")
print(f"LLM Judge: {settings.llm_judge}")
print(f"LLM Generator: {settings.llm_generator}")
print()
print("=== Latency Optimizations ===")
print(f"Top K Dense: {settings.top_k_dense} (should be 8)")
print(f"Top K Sparse: {settings.top_k_sparse} (should be 8)")
print(f"Top K Rerank: {settings.top_k_rerank} (should be 3)")
print()
print("=== Caching ===")
print(f"Query Cache: {settings.enable_query_cache}")
print(f"Embedding Cache: {settings.enable_embedding_cache}")
print(f"Cache TTL: {settings.cache_ttl_sec}s")
EOF
```

### Expected Output:
```
=== Model Configuration ===
Embedding Model: all-minilm:latest
LLM Router: gemma3:1b
LLM Judge: qwen:1.8b
LLM Generator: gemma3:1b

=== Latency Optimizations ===
Top K Dense: 8 (should be 8)
Top K Sparse: 8 (should be 8)
Top K Rerank: 3 (should be 3)

=== Caching ===
Query Cache: True
Embedding Cache: True
Cache TTL: 3600s
```

---

## 💬 Full User Flow Test

### Step 1: Upload Documents

```bash
# Create a simple test file
echo "This is a test document about Python programming. Python is great." > test.txt

# Upload it
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "files=@test.txt"
```

### Step 2: Verify Document Listed

```bash
curl http://localhost:8000/documents | jq '.documents'
```

### Step 3: Query All Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "stream": false}' | jq .
```

### Step 4: Query Specific Document

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "documents": ["test.txt"], "stream": false}' | jq .
```

### Step 5: Verify Cache Hit

```bash
# Same query again - should be instant
time curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?", "stream": false}' | jq '.latency_ms'
```

---

## 🎯 Streamlit UI Features

### Available in Sidebar

- ✅ **Upload Documents** - Drag & drop or select files
- ✅ **Ingested Files** - Shows successfully imported files
- ✅ **Available Documents** - All ingested documents with chunk counts
- ✅ **Select documents to query** - Multiselect for filtering
- ✅ **Session Stats** - Queries count and average latency
- ✅ **Quick Examples** - Pre-built example queries

### Chat Features

- ✅ **Real-time responses** - SSE streaming mode available
- ✅ **Quality metrics** - Faithfulness, relevancy, precision, hallucination
- ✅ **Source citations** - Click to expand and see which chunks were used
- ✅ **Response copy** - Copy answers to clipboard

---

## 🐛 Debugging

### Check if Models Are Loaded

```bash
curl http://localhost:8000/health | jq '.components.models'
```

### Monitor API Logs

```bash
# Terminal with API running
tail -f see logs for errors

# Should see MODEL LOADED messages
# LOG: Model loaded: gemma3:1b
# LOG: Model loaded: qwen:1.8b
```

### Check Cache Status

```bash
python3 << 'EOF'
from src.utils.cache import get_cache_manager
import os

cache = get_cache_manager()
cache_files = list(cache.cache_dir.glob("*.json"))
print(f"Cache files: {len(cache_files)}")
print(f"Cache dir: {cache.cache_dir}")
print(f"Cache TTL: {cache.ttl_sec}s")

# Show first few files
for f in cache_files[:5]:
    size_kb = os.path.getsize(f) / 1024
    print(f"  {f.name} ({size_kb:.1f} KB)")
EOF
```

### Verify Document Filtering Works

```bash
# List all docs
curl http://localhost:8000/documents | jq '.documents[].name'

# Query with non-existent doc (should still work, returns empty filter)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "documents": ["nonexistent.pdf"], "stream": false}' \
  -s | jq '.error'

# Should be null (no error) - the pipeline handles gracefully
```

---

## 📊 Performance Baseline

Run this to get performance metrics:

```bash
python3 << 'EOF'
import httpx
import time

api = "http://localhost:8000"

print("=== Performance Test ===\n")

# Test 1: First query (uncached)
print("1️⃣  First query (uncached)...")
t0 = time.time()
r = httpx.post(f"{api}/query", json={"query": "test query", "stream": False}, timeout=120)
elapsed1 = (time.time() - t0) * 1000
latency1 = r.json().get("latency_ms", 0)
print(f"   Total: {elapsed1:.0f}ms | API latency: {latency1:.0f}ms\n")

# Test 2: Second query (cached)
print("2️⃣  Same query (cached)...")
t0 = time.time()
r = httpx.post(f"{api}/query", json={"query": "test query", "stream": False}, timeout=120)
elapsed2 = (time.time() - t0) * 1000
latency2 = r.json().get("latency_ms", 0)
print(f"   Total: {elapsed2:.0f}ms | API latency: {latency2:.0f}ms")
print(f"   Speedup: {elapsed1/elapsed2:.1f}x faster ✨\n")

# Test 3: Multi-document query
print("3️⃣  Multi-document query...")
docs = ["doc1", "doc2", "doc3"]
t0 = time.time()
r = httpx.post(f"{api}/query", 
    json={"query": "test", "documents": docs, "stream": False}, 
    timeout=120)
if r.status_code == 200:
    elapsed3 = (time.time() - t0) * 1000
    latency3 = r.json().get("latency_ms", 0)
    print(f"   Total: {elapsed3:.0f}ms | API latency: {latency3:.0f}ms")
else:
    print(f"   Note: No documents found (expected on first run)\n")

print("✅ Performance test complete!")
EOF
```

---

## 🎉 All Tests Passing?

If all above tests pass:

✅ File upload working  
✅ Document listing working  
✅ Document filtering working  
✅ Caching working (huge speedup)  
✅ Models configured correctly  
✅ Latency optimized  

**You're ready to go!** 🚀

Start using the system:

1. Open **http://localhost:8501** (Streamlit UI)
2. Upload documents in the sidebar
3. Select specific documents or query all
4. Enjoy low-latency responses with caching!

---

## 📞 Need Help?

### Check Logs

```bash
# API logs
tail -n 100 -f <terminal with make run>

# Streamlit logs  
tail -n 100 -f <terminal with streamlit run app.py>
```

### Clear Everything and Restart

```bash
# Stop all services (Ctrl+C in terminals)

# Clear cache
rm -rf data/cache/*

# Clear ChromaDB
rm -rf data/chroma/*

# Restart
make up
make run
streamlit run app.py
```

### Check Model Availability

```bash
ollama list

# Should see:
# all-minilm:latest
# gemma3:1b
# qwen:1.8b
```

---

**Happy RAG-ing!** 🎉
