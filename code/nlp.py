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

    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "").strip()
        metadata = chunk.get("metadata", {})
        url = metadata.get("source_url", "").strip()
        title = metadata.get("title", "").strip()

        if url and url not in seen_urls:
            seen_urls[url] = title or url

        context_parts.append(f"[Chunk {i}]\nSource: {url}\nTitle: {title}\n{text}")

    context_block = "\n\n".join(context_parts)

    source_list = "\n".join(
        f"- {title}: {url}" for url, title in seen_urls.items()
    )

    prompt = f"""You are a veterinary parasitology assistant for CAPC (Companion Animal Parasite Council).
Answer the user's question using ONLY the context chunks provided below.
Do not use any knowledge from outside these chunks.

Output format (follow this structure when the chunks contain relevant guidelines):
1. Start with: "Based on the available CAPC guidelines, here are the recommendations for [parasite/condition name] ([scientific name if applicable]) [prevention/treatment/etc.] in [species if relevant]:"
2. Use these section headers when the context supports them; use bullet points under each:
   - Screening Protocols:
   - Prevention and Prophylaxis:
   - Treatment Strategies:
   - Additional Recommendations: (e.g. external links like petdiseasealerts.org if mentioned)
3. If the context is limited, add a brief "Note:" explaining what is or is not covered.
4. End with "Sources:" on its own line, then list each relevant CAPC URL as "- https://capcvet.org/..."

Rules:
- Include only sections for which the chunks provide information; omit empty sections.
- Use bullet points (- item) under each section for clarity.
- If the chunks do not contain enough information to answer the question, say exactly:
  "The available CAPC guidelines do not contain enough information to answer this question."
  then add a Note if helpful, then the Sources section.
- Do not speculate or add information not present in the chunks.
- List every relevant source URL from the chunks in the Sources section.

User question: {query}

Context chunks:
{context_block}

Available sources:
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