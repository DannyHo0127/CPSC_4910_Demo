# =============================================================================
# retrieval.py
# Hybrid RAG Retrieval — SageMaker AI Jupyter
#
# Required pip installs (run once in terminal):
#   pip install boto3 chromadb sentence-transformers rank-bm25
#   pip install torch transformers accelerate numpy tqdm
# =============================================================================

import os
import json
import logging
import numpy as np
import torch
import boto3
import chromadb
from chromadb.config import Settings
from pathlib import Path
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrieval")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
S3_BUCKET = os.getenv("CAPC_S3_BUCKET", "capc-data")
S3_JSON_PREFIX = os.getenv("CAPC_S3_JSON_PREFIX", "data/guidelines/")
S3_CHROMA_PREFIX = os.getenv("CAPC_S3_CHROMA_PREFIX", "vector-db/chroma-db/")
RERANK_MODEL_ID = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL_ID = "BAAI/bge-m3"
CHROMA_LOCAL_DIR = "/tmp/chroma_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RRF_K = 60


# =============================================================================
# 1. Load all JSON chunks from S3
# =============================================================================

def load_json_from_s3(bucket: str = S3_BUCKET, prefix: str = S3_JSON_PREFIX) -> list[dict]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    sparse_collection: list[dict] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                chunks = json.loads(body.decode("utf-8"))
                for chunk in chunks:
                    chunk["_s3_key"] = key
                sparse_collection.extend(chunks)
            except Exception as exc:
                logger.warning("Skipping %s — %s", key, exc)

    logger.info("Loaded %d chunks from s3://%s/%s", len(sparse_collection), bucket, prefix)
    return sparse_collection


# =============================================================================
# 2. BM25 keyword search over JSON sparse_collection
# =============================================================================

def keyword_search(query: str, sparse_collection: list[dict], top_k: int = 10) -> list[dict]:
    texts = [f"{c.get('title', '')} {c.get('text', '')}".strip() for c in sparse_collection]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict] = []
    for i in top_indices:
        c = sparse_collection[i]
        results.append({
            "id": str(c.get("id", i)),
            "text": texts[i],
            "metadata": {
                "title": c.get("title", ""),
                "source_url": c.get("source_url", ""),
                "s3_key": c.get("_s3_key", ""),
            },
            "score": float(scores[i]),
        })

    logger.info("BM25 returned %d results for query: %r", len(results), query)
    return results


# =============================================================================
# 3. Load ChromaDB from S3 (sync to local, return collection)
# =============================================================================

def load_chromadb_from_s3(
    bucket: str = S3_BUCKET,
    s3_prefix: str = S3_CHROMA_PREFIX,
    local_dir: str = CHROMA_LOCAL_DIR,
) -> chromadb.Collection:
    """
    Mirror the ChromaDB directory from S3 to `local_dir` and return
    the first available collection — no collection name needs to be known
    in advance; it is discovered automatically from the database.

    Only downloads files that do not already exist locally.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(s3_prefix):]
            if not relative:
                continue
            local_path = Path(local_dir) / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                s3.download_file(bucket, key, str(local_path))

    logger.info("ChromaDB synced to %s", local_dir)

    client = chromadb.PersistentClient(
        path=local_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    collections = client.list_collections()
    if not collections:
        raise RuntimeError("No collections found in the synced ChromaDB.")

    dense_collection = client.get_collection(collections[0].name)
    logger.info(
        "Auto-discovered dense_collection: '%s' (%d embeddings)",
        collections[0].name,
        dense_collection.count(),
    )
    return dense_collection


# =============================================================================
# 3b. Load ChromaDB from local path (for prototype without S3)
# =============================================================================

def load_chromadb_local(path: str) -> chromadb.Collection:
    """
    Load ChromaDB from a local directory (e.g. the project 'code' folder).
    Returns the first available collection.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Chroma path not found: {path}")

    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )

    collections = client.list_collections()
    if not collections:
        raise RuntimeError("No collections found in the local ChromaDB.")

    dense_collection = client.get_collection(collections[0].name)
    logger.info(
        "Local dense_collection: '%s' (%d embeddings)",
        collections[0].name,
        dense_collection.count(),
    )
    return dense_collection


# =============================================================================
# 3c. Build sparse corpus from Chroma collection (for local-only prototype)
# =============================================================================

def build_sparse_from_chroma(dense_collection: chromadb.Collection) -> list[dict]:
    """
    Export all documents from a Chroma collection into the sparse_collection
    format (list of dicts with id, title, text, source_url) for BM25 keyword search.
    Use when S3 JSON is not available (e.g. local prototype).
    """
    data = dense_collection.get(
        include=["documents", "metadatas"],
    )
    ids = data["ids"] or []
    documents = data["documents"] or []
    metadatas = data["metadatas"] or []

    sparse_collection: list[dict] = []
    for i, (doc_id, text) in enumerate(zip(ids, documents)):
        meta = metadatas[i] if i < len(metadatas) else {}
        sparse_collection.append({
            "id": doc_id,
            "title": meta.get("title") or meta.get("source_title") or "",
            "text": text or "",
            "source_url": meta.get("source_url") or meta.get("url") or "",
        })
    logger.info("Built sparse corpus from Chroma: %d chunks", len(sparse_collection))
    return sparse_collection


# =============================================================================
# 4. Dense vector search on ChromaDB (L2 distance)
# =============================================================================

def dense_vector_search(
    query: str,
    dense_collection: chromadb.Collection,
    top_k: int = 10,
    embed_model_id: str = EMBED_MODEL_ID,
) -> list[dict]:
    """
    Embed `query` using BAAI/bge-m3 then search the ChromaDB collection,
    returning top_k results.

    NOTE: This collection was built with L2 (Euclidean) distance — ChromaDB will
    use L2 automatically based on how the collection was originally configured.
    Scores returned are L2 distances (lower = more similar).
    """
    embed_model = SentenceTransformer(embed_model_id, device=DEVICE)
    embedded_query = embed_model.encode(query, normalize_embeddings=True).tolist()

    results = dense_collection.query(
        query_embeddings=[embedded_query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": float(results["distances"][0][i]),
        })

    logger.info("Dense vector search returned %d hits", len(hits))
    return hits


# =============================================================================
# 5. Hybrid retrieve
# =============================================================================

def hybrid_retrieve(
    query_text: str,
    sparse_collection: list[dict],
    dense_collection: chromadb.Collection,
    top_k: int = 10,
) -> list[dict]:

    dense_hits = dense_vector_search(query_text, dense_collection, top_k)
    sparse_hits = keyword_search(query_text, sparse_collection, top_k)

    id_to_chunk: dict[str, dict] = {}
    id_to_score: dict[str, float] = {}

    for ranked_list in (dense_hits, sparse_hits):
        for rank, hit in enumerate(ranked_list, start=1):
            hit_id = hit["id"]
            id_to_score[hit_id] = id_to_score.get(hit_id, 0.0) + 1.0 / (RRF_K + rank)
            id_to_chunk[hit_id] = hit

    fused_ids = sorted(id_to_chunk.keys(), key=lambda x: id_to_score[x], reverse=True)
    candidates = [
        dict(id_to_chunk[hid], rrf_score=id_to_score[hid])
        for hid in fused_ids[:top_k]
    ]

    return candidates


# =============================================================================
# 6. Reranking with BAAI/bge-reranker-v2-m3
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    model_id: str = RERANK_MODEL_ID,
    batch_size: int = 16,
) -> list[dict]:
    
    if not candidates:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    model.to(DEVICE)

    texts = [c["text"] for c in candidates]
    scores: list[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            pairs = [[query, t] for t in batch]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**inputs).logits.squeeze(-1)
            batch_scores = torch.sigmoid(logits).cpu().numpy().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

    ranked = [dict(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    logger.info("Reranker returning top %d of %d candidates", top_n, len(candidates))
    return ranked[:top_n]


# =============================================================================
# 7. Hybrid retrieve and rerank — main entry point called by other files
# =============================================================================

def hybrid_retrieve_and_rerank(
    query_text: str,
    sparse_collection: list[dict],
    dense_collection: chromadb.Collection,
    hybrid_top_k: int = 10,
    rerank_top_n: int = 5,
) -> list[dict]:

    candidates = hybrid_retrieve(query_text, sparse_collection, dense_collection, hybrid_top_k)

    return rerank(query_text, candidates, top_n=rerank_top_n)
