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
        no_results_msg = ("I'm sorry, but I do not have any relevant articles in the current corpus on this topic. "
                          "This does not imply that none exist. You may try expanding your search or rephrasing your query.")
        return {
            "answer": translate_from_english(no_results_msg, source_lang),
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


def compare_papers(papers: list[dict], query: str, model: str = None) -> str:
    """Compare retrieved papers against the user's research direction using a structured markdown table."""
    if not papers:
        return ""
    context = "\n".join(
        f'- "{p["title"]}" ({p["year"]}): {p["abstract"][:300]}' for p in papers
    )
    system_prompt = (
        "Compare the following academic papers against the user's research direction. "
        "Produce a markdown table with these columns:\n"
        "| Paper | Methodology | Datasets | Key Contribution | Relation to Your Direction |\n"
        "Fill every cell concisely. After the table, add a brief paragraph summarizing the most important "
        "differences and potential opportunities. ONLY reference papers provided. NEVER fabricate citations. "
        "Respond in the same language as the research direction."
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def state_of_the_art_review(papers: list[dict], query: str, model: str = None) -> str:
    """Produce a structured state-of-the-art review: thematic groups, field evolution, and dominant trends."""
    if not papers:
        return ""
    context = "\n".join(
        f'- "{p["title"]}" ({p["year"]}): {p["abstract"][:300]}' for p in papers
    )
    system_prompt = (
        "You are an academic research assistant. Given the papers below and the user's research direction, "
        "produce a structured state-of-the-art review in markdown.\n\n"
        "1. **Thematic Groups**: Organize the papers into coherent thematic groups or research approaches. "
        "Give each group a descriptive heading and list the papers that belong to it.\n"
        "2. **Field Evolution**: Using the publication years, describe how the field has evolved over time — "
        "what came first, what shifted, and where the field is heading.\n"
        "3. **Dominant Trends**: Identify the dominant methodologies, datasets, or paradigms that appear most frequently.\n\n"
        "ONLY reference papers provided. NEVER fabricate citations. "
        "Respond in the same language as the research direction."
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def detect_research_gaps(papers: list[dict], query: str, model: str = None) -> str:
    """Analyze retrieved papers and identify unexplored areas, method gaps, and research opportunities."""
    if not papers:
        return ""
    context = "\n".join(
        f'- "{p["title"]}" ({p["year"]}): {p["abstract"][:300]}' for p in papers
    )
    system_prompt = (
        "You are an academic research assistant. Given the papers below and the user's research direction, "
        "identify research gaps in markdown.\n\n"
        "1. **Uncovered Aspects**: Identify what aspects of the user's research direction are NOT addressed "
        "by the existing papers.\n"
        "2. **Unexplored Combinations**: Suggest specific combinations of methods, datasets, or approaches "
        "that none of the papers have tried but that could be promising.\n"
        "3. **Limitations & Opportunities**: Highlight concrete limitations in the current work that create "
        "opportunities for new research.\n\n"
        "Be specific and actionable. ONLY reference papers provided. NEVER fabricate citations. "
        "Respond in the same language as the research direction."
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)



def generate_bibtex(papers: list[dict]) -> str:
    """Generate BibTeX entries for retrieved papers."""
    entries = []
    for p in papers:
        # Use arxiv ID as citation key, sanitize for BibTeX
        key = p["id"].replace("/", "_").replace(".", "_")
        first_author = p["authors"].split(",")[0].strip().split()[-1] if p["authors"] else "Unknown"
        entry = (
            f"@article{{{first_author}{p['year']}_{key},\n"
            f"  title     = {{{p['title']}}},\n"
            f"  author    = {{{p['authors']}}},\n"
            f"  year      = {{{p['year']}}},\n"
            f"  eprint    = {{{p['id']}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  url       = {{{p['arxiv_url']}}}\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)