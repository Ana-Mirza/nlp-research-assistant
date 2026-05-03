"""Streamlit frontend for the Academic Research Assistant."""

import streamlit as st
from rag import research_assistant, summarize_paper, compare_papers, state_of_the_art_review, detect_research_gaps, generate_bibtex
from retrieval import compute_cosine_similarities
from config import AVAILABLE_MODELS, DEFAULT_MODEL

st.set_page_config(page_title="Academic Research Assistant", layout="wide", page_icon="\U0001f4da")

# UI labels per language (fallback to English)
UI_LABELS = {
    "en": {"analysis": "Analysis", "papers": "Retrieved Papers", "summary": "Summary",
           "generating": "⏳ Generating summary...", "compare": "Compare Papers",
           "classify": "Classify Papers by Methodology", "sota": "State-of-the-Art Review",
           "gaps": "Detect Research Gaps", "search": "Search", "direction": "Research direction",
           "placeholder": "e.g., I am researching implementations of federated neural topic models"},
    "es": {"analysis": "Análisis", "papers": "Artículos encontrados", "summary": "Resumen",
           "generating": "⏳ Generando resumen...", "compare": "Comparar artículos",
           "classify": "Clasificar por metodología", "sota": "Revisión del estado del arte",
           "gaps": "Detectar brechas de investigación", "search": "Buscar", "direction": "Dirección de investigación",
           "placeholder": "ej., Estoy investigando modelos de lenguaje grandes para diagnóstico médico"},
    "fr": {"analysis": "Analyse", "papers": "Articles trouvés", "summary": "Résumé",
           "generating": "⏳ Génération du résumé...", "compare": "Comparer les articles",
           "classify": "Classifier par méthodologie", "sota": "Revue de l'état de l'art",
           "gaps": "Détecter les lacunes", "search": "Rechercher", "direction": "Direction de recherche",
           "placeholder": "ex., Je recherche des modèles de langage pour le diagnostic médical"},
    "ro": {"analysis": "Analiză", "papers": "Articole găsite", "summary": "Rezumat",
           "generating": "⏳ Se generează rezumatul...", "compare": "Compară articolele",
           "classify": "Clasifică după metodologie", "sota": "Revizuirea stadiului actual",
           "gaps": "Detectează lacune de cercetare", "search": "Caută", "direction": "Direcția de cercetare",
           "placeholder": "ex., Vreau să studiez modele neurale pentru procesarea limbajului natural"},
    "de": {"analysis": "Analyse", "papers": "Gefundene Artikel", "summary": "Zusammenfassung",
           "generating": "⏳ Zusammenfassung wird erstellt...", "compare": "Artikel vergleichen",
           "classify": "Nach Methodik klassifizieren", "sota": "Stand der Technik",
           "gaps": "Forschungslücken erkennen", "search": "Suchen", "direction": "Forschungsrichtung",
           "placeholder": "z.B., Ich erforsche neuronale Netze für medizinische Diagnose"},
}

def _ui(key):
    """Get UI label for current language."""
    lang = st.session_state.get("ui_lang", "en")
    return UI_LABELS.get(lang, UI_LABELS["en"]).get(key, UI_LABELS["en"].get(key, key))

@st.cache_resource
def _preload():
    from retrieval import _get_collection, _get_bm25, _get_reranker, _get_embedder
    _get_collection()
    _get_bm25()
    _get_reranker()
    _get_embedder()

with st.spinner("Loading models..."):
    _preload()

# --- Sidebar ---
with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
    selected_temp = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)
    top_k = st.slider("Top-k papers", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        "**Academic Research Assistant** uses hybrid retrieval (dense + sparse + re-ranking) "
        "over 388K arXiv papers and an LLM to analyse research directions."
    )

# --- Main area ---
st.title("\U0001f4da Academic Research Assistance System")
st.markdown("Enter a research direction and get relevant papers with an AI-generated analysis.")

query = st.text_area(
    "Research direction",
    placeholder="e.g., I am researching implementations of federated neural topic models",
)

if st.button("\U0001f50d Search"):
    if not query.strip():
        st.warning("Please enter a research direction.")
    else:
        with st.spinner("Retrieving papers and generating analysis..."):
            result = research_assistant(query, model=selected_model, temperature=selected_temp, top_k=top_k)
            from translate import translate_to_english
            english_query, _ = translate_to_english(query)
            paper_ids = [p["id"] for p in result.get("papers", [])]
            cosine_sims = compute_cosine_similarities(english_query, paper_ids) if paper_ids else {}

        st.session_state["result"] = result
        st.session_state["cosine_sims"] = cosine_sims

        # Set UI language based on detected query language
        st.session_state["ui_lang"] = result.get("source_lang", "en")

        # Clear stale results from previous query
        for key in list(st.session_state.keys()):
            if key.startswith(("summary_", "comparison", "classification", "sota", "research_gaps")):
                del st.session_state[key]

if "result" in st.session_state:
    result = st.session_state["result"]
    cosine_sims = st.session_state.get("cosine_sims", {})
    papers = result.get("papers", [])
    answer = result.get("answer", "")

    if not papers:
        st.info(answer)
    else:
        col_analysis, col_papers = st.columns([3, 2])

        with col_analysis:
            st.subheader(f"\U0001f4dd {_ui('analysis')}")
            st.markdown(answer)

            if st.button(f"\U0001f504 {_ui('compare')}"):
                with st.spinner("Comparing papers..."):
                    comparison = compare_papers(papers, result.get("query", ""), model=result.get("model"), lang=result.get("source_lang", "en"))
                    st.session_state["comparison"] = comparison
            if "comparison" in st.session_state:
                st.subheader(f"\U0001f504 {_ui('compare')}")
                st.markdown(st.session_state["comparison"])

            if st.button(f"\U0001f3f7\ufe0f {_ui('classify')}"):
                with st.spinner("Classifying papers..."):
                    from rag import classify_papers
                    classification = classify_papers(papers, result.get("query", ""), model=result.get("model"), lang=result.get("source_lang", "en"))
                    st.session_state["classification"] = classification
            if "classification" in st.session_state:
                st.subheader(f"\U0001f3f7\ufe0f {_ui('classify')}")
                st.markdown(st.session_state["classification"])

            if st.button(f"\U0001f4da {_ui('sota')}"):
                with st.spinner("Generating state-of-the-art review..."):
                    sota = state_of_the_art_review(papers, result.get("query", ""), model=result.get("model"), lang=result.get("source_lang", "en"))
                    st.session_state["sota_review"] = sota
            if "sota_review" in st.session_state:
                st.subheader(f"\U0001f4da {_ui('sota')}")
                st.markdown(st.session_state["sota_review"])

            if st.button(f"\U0001f50e {_ui('gaps')}"):
                with st.spinner("Analyzing research gaps..."):
                    gaps = detect_research_gaps(papers, result.get("query", ""), model=result.get("model"), lang=result.get("source_lang", "en"))
                    st.session_state["research_gaps"] = gaps
            if "research_gaps" in st.session_state:
                st.subheader(f"\U0001f50e {_ui('gaps')}")
                st.markdown(st.session_state["research_gaps"])

        with col_papers:
            st.subheader(f"\U0001f4c4 {_ui('papers')} ({len(papers)})")
            for i, paper in enumerate(papers):
                cos_sim = cosine_sims.get(paper["id"], 0.0)
                with st.expander(f"**{paper['title']}** ({paper['year']})"):
                    st.markdown(f"**Authors:** {paper['authors']}")
                    st.markdown(f"**Categories:** {paper['categories']}")
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Re-ranker Score", f"{paper['score']:.3f}")
                    col_s2.metric("Cosine Similarity", f"{cos_sim:.3f}")
                    # Color-coded relevance indicator
                    s = paper['score']
                    if s >= 7.0:
                        st.markdown(":green[⬤ **High Relevance**]")
                    elif s >= 4.0:
                        st.markdown(":orange[⬤ **Medium Relevance**]")
                    else:
                        st.markdown(":blue[⬤ **Low Relevance**]")
                    source_label = "arXiv" if paper.get("source", "arxiv") == "arxiv" else "PubMed"
                    paper_url = paper.get("url", paper.get("arxiv_url", "#"))
                    st.markdown(f"[\U0001f4ce {source_label} Link]({paper_url})")
                    st.markdown(f"**Abstract:** {paper['abstract'][:500]}{'...' if len(paper['abstract']) > 500 else ''}")

                    # Summary as collapsible section (progressively loaded)
                    summary_key = f"summary_{paper['id']}"
                    with st.expander("\U0001f4cb Summary"):
                        if summary_key in st.session_state:
                            st.markdown(st.session_state[summary_key])
                        else:
                            st.caption(_ui("generating"))

        # BibTeX export
        if papers:
            bibtex = generate_bibtex(papers)
            st.download_button(
                "\U0001f4e5 Export BibTeX",
                data=bibtex,
                file_name="papers.bib",
                mime="text/plain",
            )

        st.caption(f"\u23f1\ufe0f Response time: {result.get('time_seconds', 0):.2f}s | Model: {result.get('model', 'N/A')} | Language: {result.get('source_lang', 'en')}")

        # Progressive summary generation: generate one missing summary per render cycle
        # This runs AFTER all results are displayed, so the UI is never blocked
        if papers:
            for p in papers:
                sk = f"summary_{p['id']}"
                if sk not in st.session_state:
                    st.session_state[sk] = summarize_paper(p, model=result.get("model"))
                    st.rerun()  # Rerun to display the newly generated summary
                    break  # Only one per cycle to keep UI responsive
