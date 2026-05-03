"""Streamlit frontend for the Academic Research Assistant."""

import streamlit as st
from rag import research_assistant, summarize_paper
from retrieval import compute_cosine_similarities
from config import AVAILABLE_MODELS, DEFAULT_MODEL

st.set_page_config(page_title="Academic Research Assistant", layout="wide", page_icon="📚")

# Preload all models on startup so first query is fast
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
    st.header("⚙️ Settings")
    selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
    selected_temp = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)
    top_k = st.slider("Top-k papers", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        "**Academic Research Assistant** uses hybrid retrieval (dense + sparse + re-ranking) "
        "over 388K arXiv papers and an LLM to analyse research directions."
    )

# --- Main area ---
st.title("📚 Academic Research Assistance System")
st.markdown("Enter a research direction and get relevant papers with an AI-generated analysis.")

query = st.text_area(
    "Research direction",
    placeholder="e.g., I am researching implementations of federated neural topic models",
)

if st.button("🔍 Search"):
    if not query.strip():
        st.warning("Please enter a research direction.")
    else:
        with st.spinner("Retrieving papers and generating analysis..."):
            result = research_assistant(query, model=selected_model, temperature=selected_temp, top_k=top_k)
            # Compute cosine similarities using English query (papers are in English)
            from translate import translate_to_english
            english_query, _ = translate_to_english(query)
            paper_ids = [p["id"] for p in result.get("papers", [])]
            cosine_sims = compute_cosine_similarities(english_query, paper_ids) if paper_ids else {}

        st.session_state["result"] = result
        st.session_state["cosine_sims"] = cosine_sims

# Display results from session state
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
            st.subheader("📝 Analysis")
            st.markdown(answer)

            # Classification section — on-demand to keep main query fast
            if st.button("🏷️ Classify Papers by Methodology"):
                with st.spinner("Classifying papers..."):
                    from rag import classify_papers
                    from translate import translate_to_english, translate_from_english
                    english_query, source_lang = translate_to_english(result.get("query", ""))
                    classification = classify_papers(papers, english_query, model=result.get("model"))
                    classification = translate_from_english(classification, source_lang)
                    st.session_state["classification"] = classification
            if "classification" in st.session_state:
                st.subheader("🏷️ Paper Classification by Methodology")
                st.markdown(st.session_state["classification"])

        with col_papers:
            st.subheader(f"📄 Retrieved Papers ({len(papers)})")
            for i, paper in enumerate(papers):
                cos_sim = cosine_sims.get(paper["id"], 0.0)
                with st.expander(f"**{paper['title']}** ({paper['year']})"):
                    st.markdown(f"**Authors:** {paper['authors']}")
                    st.markdown(f"**Categories:** {paper['categories']}")
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Re-ranker Score", f"{paper['score']:.3f}")
                    col_s2.metric("Cosine Similarity", f"{cos_sim:.3f}")
                    st.markdown(f"[📎 arXiv Link]({paper['arxiv_url']})")
                    st.markdown(f"**Abstract:** {paper['abstract'][:500]}{'...' if len(paper['abstract']) > 500 else ''}")

                    # Per-paper summary button
                    summary_key = f"summary_{paper['id']}"
                    if st.button(f"📋 Generate Summary", key=f"btn_{paper['id']}"):
                        with st.spinner("Generating summary..."):
                            st.session_state[summary_key] = summarize_paper(paper, model=result.get("model"))
                    if summary_key in st.session_state:
                        st.info(st.session_state[summary_key])

        st.caption(f"⏱️ Response time: {result.get('time_seconds', 0):.2f}s | Model: {result.get('model', 'N/A')} | Language: {result.get('source_lang', 'en')}")
