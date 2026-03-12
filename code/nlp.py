# =============================================================================
# nlp.py
# NLP layer for the RAG pipeline — query enhancement and response generation
#
# Depends on retrieval.py being present in the same directory.
# Uses Llama 3.1 8B running locally via Ollama for all LLM calls.
#
# Required setup (run once in terminal):
#   pip install ollama
#   curl -fsSL https://ollama.com/install.sh | sh
#   ollama pull llama3.1:8b
# =============================================================================

import logging
import ollama
import retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nlp")

OLLAMA_MODEL = "llama3.1:8b"


# =============================================================================
# 1. Query enhancement
# =============================================================================

def enhance_query(query: str) -> str:
    prompt = f"""You are a veterinary parasitology expert helping to optimise search queries
for a retrieval system containing CAPC (Companion Animal Parasite Council) guidelines.
The database contains clinical and epidemiological information about animal parasites
including life cycles, disease, diagnosis, treatment, prevention, prevalence, host
associations, and public health considerations.

Your task: rewrite the user query below into an enhanced search query that will maximise
recall from both keyword (BM25) and semantic (vector) search. Follow these rules:
- Expand abbreviations and colloquial terms into proper veterinary/scientific terminology
- Add relevant species context if the query implies a host animal (dog, cat, etc.)
- Include synonyms or related terms for the core parasite or condition
- Keep the enhanced query concise — one to three sentences maximum
- Output only the enhanced query text, nothing else

User query: {query}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    enhanced = response["message"]["content"].strip()
    logger.info("Enhanced query: %r", enhanced)
    return enhanced


# =============================================================================
# 2. Response generation from retrieved chunks
# =============================================================================

def generate_response(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "No relevant information was found in the CAPC guidelines for your query. "
            "Please try rephrasing your question or visit https://capcvet.org for full guidelines."
        )

    # Build context block from chunks, deduplicating source URLs as we go
    context_parts = []
    seen_urls = {}

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})
        url = metadata.get("source_url", "").strip()
        title = metadata.get("title", "").strip()

        if url and url not in seen_urls:
            seen_urls[url] = title or url

        context_parts.append(f"Source: {url}\nTitle: {title}\n{text}")

    context_block = "\n\n---\n\n".join(context_parts)

    source_list = "\n".join(
        f"- {title}: {url}" for url, title in seen_urls.items()
    )

    prompt = f"""You are a veterinary parasitology assistant for CAPC (Companion Animal Parasite Council).
Answer the user's question using ONLY the context provided below.
Do not use any knowledge from outside this context.

Rules:
- Answer in a clear, detailed manner based on what the context actually says. Use whatever structure fits the question (prose, lists, or sections as needed).
- Do NOT mention "chunk", "Chunk", or any chunk numbers in your response. Write as if you are summarizing the guidelines directly.
- If the context does not contain enough information to answer the question, say so clearly, then summarize what was found.
- Do not speculate or add information not present in the context.
- Always end with a "Sources:" section. List ONLY the CAPC URLs that were directly used to answer the question (omit sources that are not relevant to the user's question). Each source once, with full URL, e.g. "- https://capcvet.org/guidelines/..."

User question: {query}

Context:
{context_block}

Available sources (use only those relevant to your answer):
{source_list}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"].strip()
    logger.info("Response generated from %d chunks", len(chunks))
    return answer


# =============================================================================
# 3. End-to-end pipeline — main entry point called by other files
# =============================================================================

def answer(
    query: str,
    sparse_collection: list[dict],
    dense_collection,
    hybrid_top_k: int = 10,
    rerank_top_n: int = 5,
) -> str:
    """
    Full NLP pipeline: enhance query → hybrid retrieve and rerank → generate response.

    Parameters
    ----------
    query : Plain-text question from the user.
    sparse_collection : JSON corpus returned by retrieval.load_json_from_s3().
    dense_collection : ChromaDB Collection returned by retrieval.load_chromadb_from_s3().
    hybrid_top_k : Number of candidates passed to the hybrid retrieval step.
    rerank_top_n : Number of reranked chunks passed to response generation.

    Returns
    -------
    A grounded answer string with links to relevant CAPC source pages.
    """
    logger.info("Received query: %r", query)

    enhanced = enhance_query(query)

    chunks = retrieval.hybrid_retrieve_and_rerank(
        query_text=enhanced,
        sparse_collection=sparse_collection,
        dense_collection=dense_collection,
        hybrid_top_k=hybrid_top_k,
        rerank_top_n=rerank_top_n,
    )

    return generate_response(query, chunks)