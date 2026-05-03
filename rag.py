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
    # Filter out individually irrelevant papers (score < 3.0)
    papers = [p for p in papers if p["score"] >= 3.0]

    if not papers:
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
        f'URL: {p["url"]}'
        for i, p in enumerate(papers, 1)
    )

    # Map language code to name for explicit LLM instruction
    lang_instruction = _lang_instruction(query, lang=source_lang)

    system_prompt = (
        "You are an academic research assistant. Your task is to help researchers "
        "understand how their research direction relates to existing work.\n\n"
        "RULES:\n"
        "- ONLY reference papers provided in the context below. NEVER fabricate paper titles, authors, or citations.\n"
        "- Identify and explain key differences and similarities between the user's research direction and the retrieved papers.\n"
        "- Organize your response as: (1) Overview of relevant papers found, (2) Key differences with the user's direction, (3) Potential gaps or opportunities.\n"
        "- Be concise. Limit your response to 300 words maximum.\n"
        "- If the provided papers are not relevant to the query, say so explicitly.\n"
        "- Paper titles may remain in their original language."
        f"{lang_instruction}"
    )

    # Pass original (non-English) query so LLM sees the user's language and responds in it
    lang_reminder = f"\n\nIMPORTANT: Write your ENTIRE response in the same language as my research direction above. Do NOT mix languages." if source_lang != "en" else ""
    user_prompt = f"My research direction: {query}\n\nRetrieved papers:\n{context}{lang_reminder}"

    answer = generate(user_prompt, system_prompt=system_prompt, model=model, temperature=temperature)

    return {
        "answer": answer,
        "papers": papers,
        "query": query,
        "source_lang": source_lang,
        "model": model,
        "time_seconds": round(time.time() - start, 2),
    }


def _lang_instruction(query: str, lang: str = None) -> str:
    """Return explicit LLM language instruction like ' You MUST respond in Romanian.'"""
    if lang is None:
        from translate import detect_language
        lang = detect_language(query)
    if lang == "en":
        return ""
    names = {
        "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
        "pt": "Portuguese", "ro": "Romanian", "nl": "Dutch", "pl": "Polish",
        "ru": "Russian", "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean",
        "ar": "Arabic", "hi": "Hindi", "tr": "Turkish", "cs": "Czech",
        "hu": "Hungarian", "uk": "Ukrainian", "sv": "Swedish", "ca": "Catalan",
        "da": "Danish", "fi": "Finnish", "no": "Norwegian", "hr": "Croatian",
        "bg": "Bulgarian", "el": "Greek", "he": "Hebrew",
    }
    return f" You MUST respond in {names.get(lang, lang)}."


def classify_papers(papers: list[dict], query: str, model: str = None, lang: str = None) -> str:
    """Classify retrieved papers by methodology using LLM."""
    if not papers:
        return ""
    context = "\n".join(f'- "{p["title"]}": {p["abstract"]}' for p in papers)
    system_prompt = (
        "Classify the following academic papers into categories based on their methodological approach "
        "(e.g., supervised, unsupervised, reinforcement learning, theoretical, survey, etc.). "
        "Be concise. Use bullet points." + _lang_instruction(query, lang=lang)
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def summarize_paper(paper: dict, model: str = None) -> str:
    """Generate a 2-3 sentence summary of a paper's key contributions."""
    system_prompt = "Summarize this academic paper in 2-3 sentences, highlighting key contributions and methodology. Be concise."
    user_prompt = f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract: {paper['abstract']}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def compare_papers(papers: list[dict], query: str, model: str = None, lang: str = None) -> str:
    """Compare retrieved papers against the user's research direction using a structured markdown table."""
    if not papers:
        return ""
    context = "\n".join(
        f'- "{p["title"]}" ({p["year"]}): {p["abstract"]}' for p in papers
    )
    system_prompt = (
        f"Compare the following {len(papers)} academic paper(s) against the user's research direction. "
        f"There are EXACTLY {len(papers)} paper(s) — do NOT add any others.\n"
        "Produce a markdown table with these columns:\n"
        "| Paper | Methodology | Datasets | Key Contribution | Relation to Your Direction |\n"
        "Fill every cell concisely (max 10 words per cell). After the table, add ONE brief paragraph (max 50 words) on differences. ONLY reference papers provided. NEVER fabricate citations."
        + _lang_instruction(query, lang=lang)
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def state_of_the_art_review(papers: list[dict], query: str, model: str = None, lang: str = None) -> str:
    """Produce a structured state-of-the-art review."""
    if not papers:
        return ""
    context = "\n".join(
        f'- "{p["title"]}" ({p["year"]}): {p["abstract"]}' for p in papers
    )
    system_prompt = (
        "You are an academic research assistant. Given the papers below and the user's research direction, "
        "produce a structured state-of-the-art review in markdown.\n\n"
        "1. **Thematic Groups**: Organize the papers into coherent thematic groups.\n"
        "2. **Field Evolution**: Using publication years, describe how the field has evolved.\n"
        "3. **Dominant Trends**: Identify dominant methodologies or paradigms.\n\n"
        "ONLY reference papers provided. NEVER fabricate citations."
        + _lang_instruction(query, lang=lang)
    )
    user_prompt = f"Research direction: {query}\n\nPapers:\n{context}"
    return generate(user_prompt, system_prompt=system_prompt, model=model, temperature=0.1)


def detect_research_gaps(papers: list[dict], query: str, model: str = None, lang: str = None) -> str:
    """Identify unexplored areas and research opportunities."""
    if not papers:
        return ""
    context = "\n".join(
        f'- "{p["title"]}" ({p["year"]}): {p["abstract"]}' for p in papers
    )
    system_prompt = (
        "You are an academic research assistant. Analyze the papers below against the user's research direction "
        "and identify research gaps.\n\n"
        "You MUST use EXACTLY this markdown structure:\n\n"
        "## Uncovered Aspects\n"
        "- [bullet point for each aspect not addressed by existing papers]\n\n"
        "## Unexplored Combinations\n"
        "- [bullet point for each promising untried combination of methods/datasets]\n\n"
        "## Limitations & Opportunities\n"
        "- [bullet point for each limitation that creates an opportunity]\n\n"
        "Be specific and actionable. Keep each bullet to 1-2 sentences. "
        "ONLY reference papers provided. NEVER fabricate citations."
        + _lang_instruction(query, lang=lang)
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
            f"  url       = {{{p['url']}}}\n"
            f"}}"
        )
        entries.append(entry)
    return "\n\n".join(entries)