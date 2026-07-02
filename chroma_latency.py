#!/usr/bin/env python3

import os
import sys
import time
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

load_dotenv(os.path.join(project_root, ".env"))

try:
    from app.services.embedding import create_embedding, collection
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ── golden test query (from STT transcript) ───────────────────────────────
QUERY = "इज़ रुद्राक्ष अ गुड प्रॉब्लम सॉल्वर एंड उसने कितने क्वेश्चन करे हैं लीड कोड पे और उसे और कितने करने चाहिए कि वो अच्छा हो सके इस चीज के अंदर?"

NUM_RUNS  = 5
EMBED_BUDGET_MS = 200
CHROMA_BUDGET_MS = 20

embed_latencies  = []
chroma_latencies = []

print("=" * 60)
print(" 🔍 CHROMA + EMBEDDING LATENCY BENCHMARK")
print("=" * 60)
print(f"Query: {QUERY[:60]}...\n")

for i in range(NUM_RUNS):
    print(f"⚡ Run {i+1}/{NUM_RUNS}...")

    # 1. time embedding creation (OpenAI API call)
    start = time.perf_counter()
    embedding = create_embedding(QUERY)
    embed_ms = (time.perf_counter() - start) * 1000
    embed_latencies.append(embed_ms)

    # 2. time chroma query (local, in-memory)
    start = time.perf_counter()
    results = collection.query(query_embeddings=[embedding], n_results=3)
    chroma_ms = (time.perf_counter() - start) * 1000
    chroma_latencies.append(chroma_ms)

    print(f"   Embedding : {embed_ms:.0f}ms")
    print(f"   ChromaDB  : {chroma_ms:.1f}ms")

# ── results ───────────────────────────────────────────────────────────────
embed_avg  = sum(embed_latencies)  / len(embed_latencies)
chroma_avg = sum(chroma_latencies) / len(chroma_latencies)

print(f"\n── Embedding (OpenAI API) ───────────────────────────")
print(f"   Average : {embed_avg:.0f}ms")
print(f"   Min     : {min(embed_latencies):.0f}ms")
print(f"   Max     : {max(embed_latencies):.0f}ms")
print(f"   Budget  : {EMBED_BUDGET_MS}ms")
print(f"   Status  : {'✅ WITHIN BUDGET' if embed_avg <= EMBED_BUDGET_MS else f'❌ OVER by {embed_avg - EMBED_BUDGET_MS:.0f}ms'}")

print(f"\n── ChromaDB (local query) ───────────────────────────")
print(f"   Average : {chroma_avg:.1f}ms")
print(f"   Min     : {min(chroma_latencies):.1f}ms")
print(f"   Max     : {max(chroma_latencies):.1f}ms")
print(f"   Budget  : {CHROMA_BUDGET_MS}ms")
print(f"   Status  : {'✅ WITHIN BUDGET' if chroma_avg <= CHROMA_BUDGET_MS else f'❌ OVER by {chroma_avg - CHROMA_BUDGET_MS:.1f}ms'}")

print(f"\n── Combined RAG retrieval ───────────────────────────")
combined = embed_avg + chroma_avg
print(f"   Total   : {combined:.0f}ms")
print(f"   Budget  : 220ms")
print(f"   Status  : {'✅ WITHIN BUDGET' if combined <= 220 else f'❌ OVER by {combined - 220:.0f}ms'}")
print(f"────────────────────────────────────────────────────")

# show top result so we can verify retrieval quality
if results['documents'] and results['documents'][0]:
    print(f"\n── Top retrieved chunk (verify relevance) ──────────")
    print(f"   {results['documents'][0][0][:200]}...")
