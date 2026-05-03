"""RAG pipeline: ties retrieval + LLM generation together."""

import time

from retrieval import retrieve
from llm import generate
from translate import translate_to_english, translate_from_english
from config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


def research_assistant(query: str, model: str = None, temperature: float = None, top_k: int = 5) -> dict:
    """Answer a research query using retrieved papers + LLM generation."""
    model = model or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE

    start = time.time()

    # Translate query to English for retrieval (arXiv papers are in English)
    english_query, source_lang = translate_to_english(query)

    # Retrieve papers using English query
    papers = retrieve(english_query, top_k=top_k)

    # Check relevance threshold — cross-encoder scores typically range 0-10
    if not papers or all(p["score"] < 3.0 for p in papers):
        return {
            "answer": "I'm sorry, but I do not have any relevant articles in the current corpus on this topic. "
                      "This does not imply that none exist. You may try expanding your search or rephrasing your query.",
            "papers": [],
            "query": query,
            "model": model,
            "time_seconds": round(time.time() - start, 2),
        }

    # Build context from retrieved papers
    context = "\n\n".join(
        f'Paper {i}: "{p["title"]}" ({p["year"]})\n'
        f'Authors: {p["authors"]}\n'
        f'Abstract: {p["abstract"]}\n'
        f'URL: {p["arxiv_url"]}'
        for i, p in enumerate(papers, 1)
    )

    system_prompt = (
        "You are an academic research assistant. Your task is to help researchers "
        "understand how their research direction relates to existing work.\n\n"
        "RULES:\n"
        "- ONLY reference papers provided in the context below. NEVER fabricate paper titles, authors, or citations.\n"
        "- Identify and explain key differences and similarities between the user's research direction and the retrieved papers.\n"
        "- Organize your response as: (1) Overview of relevant papers found, (2) Key differences with the user's direction, (3) Potential gaps or opportunities.\n"
        "- If the provided papers are not relevant to the query, say so explicitly.\n"
        "- Respond in the SAME LANGUAGE as the user's query. Paper titles may remain in their original language."
    )

    user_prompt = f"My research direction: {query}\n\nRetrieved papers:\n{context}"

    answer = generate(user_prompt, system_prompt=system_prompt, model=model, temperature=temperature)

    # Translate answer back to the user's language if needed
    answer = translate_from_english(answer, source_lang)

    return {
        "answer": answer,
        "papers": papers,
        "query": query,
        "source_lang": source_lang,
        "model": model,
        "time_seconds": round(time.time() - start, 2),
    }


def classify_papers(papers: list[dict], query: str, model: str = None) -> str:
    """Classify retrieved papers by methodology using LLM."""
    if not papers:
        return ""
    context = "\n".join(f'- "{p["title"]}": {p["abstract"][:200]}' for p in papers)
    system_prompt = (
        "Classify the following academic papers into categories based on their methodological approach "
        "(e.g., supervised, unsupervised, reinforcement learning, theoretical, survey, etc.). "
        "Be concise. Use bullet points. Respond in the same language as the research direction."
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def summarize_paper(paper: dict, model: str = None) -> str:
    """Generate a 2-3 sentence summary of a paper's key contributions."""
    system_prompt = "Summarize this academic paper in 2-3 sentences, highlighting key contributions and methodology. Be concise."
    user_prompt = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract: {paper['abstract']}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)