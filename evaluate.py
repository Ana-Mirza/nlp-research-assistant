"""Evaluate the RAG system with retrieval metrics, generation metrics, and LLM-as-Judge.

Metric choices and justifications:
- Precision@K: Fraction of top-K retrieved papers that are relevant. We approximate relevance
  using the cross-encoder re-ranker score (>5.0 = relevant), since the re-ranker is trained on
  MS MARCO to predict query-document relevance. This avoids the need for manual annotations.
- MRR (Mean Reciprocal Rank): Rank of the first highly-relevant paper (score >7.0). Measures
  whether the best result appears near the top. Recommended by course slides (slide 29).
- NDCG@K: Normalized Discounted Cumulative Gain using graded relevance from re-ranker scores.
  Accounts for the position of relevant documents — a relevant paper at rank 1 is worth more
  than at rank 5. Course slides (slide 29) note this is "more informative than Recall@K when
  relevance is not binary."
- Document Coverage: % of queries where at least one paper matches topic keywords. Measures
  corpus breadth — can the system find something for any research direction?
- Faithfulness (citation accuracy): Programmatic check that paper titles mentioned in the LLM
  answer actually exist in the retrieved set. Catches hallucinated citations without needing
  ground truth. Aligned with RAGAS faithfulness concept (course slide 30).
- LLM-as-Judge: Cross-family evaluation (qwen3:8b judges llama3.1:8b outputs) to avoid
  self-evaluation bias. Scores on 1-5 scale. Score is generated BEFORE justification to avoid
  the score-then-justify antipattern (course slide 8, Turpin et al. 2023).
- Response Time: End-to-end latency. Median reported alongside mean to handle API outliers.
"""

import json
import math
import time
import re
import os
from rag import research_assistant
from llm import generate
from config import DEFAULT_MODEL, DATA_DIR

# Relevance thresholds based on cross-encoder score distribution
RELEVANT_THRESHOLD = 5.0    # Score > 5.0 = relevant (for Precision@K)
HIGH_RELEVANT_THRESHOLD = 7.0  # Score > 7.0 = highly relevant (for MRR)

TEST_QUERIES = [
    # --- Core NLP/ML (5) ---
    {"query": "federated neural topic models", "keywords": ["federated", "topic model"],
     "category": "core_nlp"},
    {"query": "knowledge distillation in large language models", "keywords": ["distillation", "language model"],
     "category": "core_nlp"},
    {"query": "zero-shot cross-lingual transfer", "keywords": ["cross-lingual", "zero-shot"],
     "category": "core_nlp"},
    {"query": "diffusion models for text generation", "keywords": ["diffusion", "text"],
     "category": "core_ml"},
    {"query": "efficient inference for large language models", "keywords": ["efficient", "inference", "language model"],
     "category": "core_ml"},
    # --- Adjacent fields (3) ---
    {"query": "attention mechanisms in computer vision", "keywords": ["attention", "vision"],
     "category": "adjacent"},
    {"query": "graph neural networks for drug discovery", "keywords": ["graph", "drug"],
     "category": "adjacent"},
    {"query": "self-supervised learning for speech recognition", "keywords": ["self-supervised", "speech"],
     "category": "adjacent"},
    # --- Weak coverage (1) ---
    {"query": "transformer architectures for protein folding", "keywords": ["transformer", "protein"],
     "category": "weak_coverage"},
    # --- Short query (1) ---
    {"query": "NAS", "keywords": ["architecture search", "neural"],
     "category": "short_query"},
    # --- Long query (1) ---
    {"query": "I am researching how to combine reinforcement learning with curriculum learning strategies to train robotic manipulation policies that generalize across different object geometries",
     "keywords": ["reinforcement", "curriculum", "robot"],
     "category": "long_query"},
    # --- Multi-language (4) ---
    {"query": "Estoy investigando modelos de lenguaje para la generación de resúmenes en español",
     "keywords": ["summarization", "language model", "generation"],
     "category": "multi_lang_es"},
    {"query": "Ich untersuche Methoden zur automatischen Erkennung von Fake News",
     "keywords": ["fake", "detection", "misinformation"],
     "category": "multi_lang_de"},
    {"query": "大規模言語モデルの効率的な推論手法について研究しています",
     "keywords": ["efficient", "inference", "language model"],
     "category": "multi_lang_ja"},
    {"query": "أبحث عن نماذج التعلم العميق لتحليل المشاعر في النصوص العربية",
     "keywords": ["sentiment", "deep learning", "emotion"],
     "category": "multi_lang_ar"},
    # --- Out of domain (4) — system should return NO papers ---
    {"query": "best recipe for chocolate cake", "keywords": [],
     "category": "out_of_domain", "expected_empty": True},
    {"query": "how to fix a leaking kitchen faucet", "keywords": [],
     "category": "out_of_domain", "expected_empty": True},
    {"query": "the impact of monetary policy on housing prices in 2024", "keywords": [],
     "category": "out_of_domain", "expected_empty": True},
    # --- Reliability check (1) ---
    {"query": "continual learning without catastrophic forgetting", "keywords": ["continual", "forgetting"],
     "category": "core_ml"},
]


# --- Retrieval Metrics ---

def precision_at_k(papers, k=5):
    """Fraction of top-K papers with cross-encoder score above relevance threshold.
    Uses re-ranker score as relevance proxy (trained on MS MARCO for relevance prediction)."""
    top_k = papers[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for p in top_k if p["score"] > RELEVANT_THRESHOLD)
    return relevant / len(top_k)


def mrr(papers):
    """Mean Reciprocal Rank: 1/rank of first highly-relevant paper.
    Returns 0 if no paper exceeds the high relevance threshold."""
    for i, p in enumerate(papers):
        if p["score"] > HIGH_RELEVANT_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(papers, k=5):
    """NDCG@K using cross-encoder scores as graded relevance.
    Scores are normalized to [0,1] by dividing by 10 (cross-encoder max ~10).
    DCG = sum(rel_i / log2(i+2)), IDCG = DCG of ideal ranking."""
    top_k = papers[:k]
    if not top_k:
        return 0.0
    rels = [min(p["score"] / 10.0, 1.0) for p in top_k]  # Normalize to [0,1]
    # DCG
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    # IDCG (ideal: sorted descending)
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def check_coverage(papers, keywords):
    """Check if at least one paper is relevant based on keyword matching in title/abstract."""
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


# --- Generation Metrics ---

def citation_faithfulness(answer, papers):
    """Check what fraction of paper titles mentioned in the answer exist in the retrieved set.
    Approximates RAGAS faithfulness by verifying citation grounding programmatically."""
    if not papers:
        return 1.0  # No papers = no citations to check
    retrieved_titles = {p["title"].lower().strip() for p in papers}
    # Extract quoted titles from the answer
    cited = re.findall(r'"([^"]{10,})"', answer)
    if not cited:
        return 1.0  # No explicit citations found
    grounded = sum(1 for c in cited if any(t in c.lower() or c.lower() in t for t in retrieved_titles))
    return grounded / len(cited)


def llm_judge_score(query, answer, papers):
    """Cross-family LLM-as-Judge: qwen3:8b rates llama3.1:8b outputs.
    Score generated FIRST (no justification) to avoid score-then-justify antipattern
    (Turpin et al. 2023, course slide 8)."""
    context = "\n".join(f'- "{p["title"]}"' for p in papers[:5])
    prompt = (
        f"Rate the following answer on a scale of 1-5 for quality as a research assistant response.\n"
        f"Consider: (1) relevance to the query, (2) grounding in the retrieved papers, (3) identification of key differences.\n\n"
        f"Query: {query}\n"
        f"Retrieved papers:\n{context}\n\n"
        f"Answer:\n{answer[:1000]}\n\n"
        f"Reply with ONLY a single number from 1 to 5."
    )
    JUDGE_MODEL = "qwen3:8b"
    resp = generate(prompt, model=JUDGE_MODEL, temperature=0.0)
    match = re.search(r'[1-5]', resp)
    return int(match.group()) if match else 3


# --- Main ---

def main():
    print(f"Evaluating RAG system with {len(TEST_QUERIES)} queries...")
    print(f"Generator: {DEFAULT_MODEL} | Judge: qwen3:8b")
    print(f"Relevance threshold: {RELEVANT_THRESHOLD} | High relevance: {HIGH_RELEVANT_THRESHOLD}")
    print("=" * 70)

    results = []
    all_times, all_precision, all_mrr, all_ndcg = [], [], [], []
    all_faithfulness, all_judge = [], []
    coverage_hits = 0
    hallucination_checks = {"total": 0, "passed": 0}

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        keywords = test["keywords"]
        category = test.get("category", "unknown")
        expected_empty = test.get("expected_empty", False)
        print(f"\n[{i+1}/{len(TEST_QUERIES)}] [{category}] {query[:80]}{'...' if len(query) > 80 else ''}")

        t0 = time.time()
        result = research_assistant(query, top_k=5)
        elapsed = time.time() - t0

        papers = result.get("papers", [])
        answer = result.get("answer", "")

        # Out-of-domain check: system should return no papers
        if expected_empty:
            hallucination_checks["total"] += 1
            refused = len(papers) == 0
            if refused:
                hallucination_checks["passed"] += 1
            print(f"  OUT-OF-DOMAIN: {'PASS (refused)' if refused else 'FAIL (returned papers)'} | Papers={len(papers)} | Time={elapsed:.1f}s")
            results.append({
                "query": query, "category": category, "num_papers": len(papers),
                "expected_empty": True, "correctly_refused": refused,
                "time_seconds": round(elapsed, 2),
                "paper_titles": [p["title"] for p in papers],
            })
            all_times.append(elapsed)
            continue

        # Retrieval metrics
        p_at_k = precision_at_k(papers, k=5)
        mrr_val = mrr(papers)
        ndcg_val = ndcg_at_k(papers, k=5)
        covered = check_coverage(papers, keywords) if keywords else len(papers) > 0
        if covered:
            coverage_hits += 1

        # Generation metrics
        faith = citation_faithfulness(answer, papers)
        judge = llm_judge_score(query, answer, papers)

        all_times.append(elapsed)
        all_precision.append(p_at_k)
        all_mrr.append(mrr_val)
        all_ndcg.append(ndcg_val)
        all_faithfulness.append(faith)
        all_judge.append(judge)

        print(f"  P@5={p_at_k:.2f} MRR={mrr_val:.2f} NDCG@5={ndcg_val:.3f} Faith={faith:.2f} Judge={judge}/5 Time={elapsed:.1f}s")

        results.append({
            "query": query,
            "category": category,
            "num_papers": len(papers),
            "coverage": covered,
            "precision_at_5": round(p_at_k, 3),
            "mrr": round(mrr_val, 3),
            "ndcg_at_5": round(ndcg_val, 3),
            "citation_faithfulness": round(faith, 3),
            "judge_score": judge,
            "time_seconds": round(elapsed, 2),
            "paper_titles": [p["title"] for p in papers],
            "paper_scores": [round(p["score"], 3) for p in papers],
        })

    # Summary — retrieval/generation metrics only for in-domain queries
    n_in_domain = len(all_precision)  # Out-of-domain queries skipped these lists
    n_total = len(TEST_QUERIES)
    n_ood = hallucination_checks["total"]
    times_sorted = sorted(all_times)
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n--- Retrieval Metrics ({n_in_domain} in-domain queries) ---")
    print(f"Document Coverage:     {coverage_hits}/{n_in_domain} ({100*coverage_hits/n_in_domain:.1f}%)")
    print(f"Precision@5 (mean):    {sum(all_precision)/n_in_domain:.3f}")
    print(f"MRR (mean):            {sum(all_mrr)/n_in_domain:.3f}")
    print(f"NDCG@5 (mean):         {sum(all_ndcg)/n_in_domain:.3f}")
    print(f"\n--- Generation Metrics ({n_in_domain} in-domain queries) ---")
    print(f"Citation Faithfulness: {sum(all_faithfulness)/n_in_domain:.3f}")
    print(f"LLM Judge (mean):      {sum(all_judge)/n_in_domain:.2f}/5  (qwen3:8b judging llama3.1:8b)")
    print(f"\n--- Hallucination Refusal ({n_ood} out-of-domain queries) ---")
    print(f"Correctly refused:     {hallucination_checks['passed']}/{n_ood} ({100*hallucination_checks['passed']/n_ood:.1f}%)" if n_ood > 0 else "N/A")
    print(f"\n--- Response Time (all {n_total} queries) ---")
    print(f"Mean:                  {sum(all_times)/n_total:.1f}s")
    print(f"Median:                {times_sorted[n_total//2]:.1f}s")
    print(f"Min / Max:             {min(all_times):.1f}s / {max(all_times):.1f}s")

    summary = {
        "retrieval": {
            "num_in_domain_queries": n_in_domain,
            "document_coverage": f"{coverage_hits}/{n_in_domain} ({100*coverage_hits/n_in_domain:.1f}%)",
            "precision_at_5": round(sum(all_precision)/n_in_domain, 3),
            "mrr": round(sum(all_mrr)/n_in_domain, 3),
            "ndcg_at_5": round(sum(all_ndcg)/n_in_domain, 3),
        },
        "generation": {
            "citation_faithfulness": round(sum(all_faithfulness)/n_in_domain, 3),
            "llm_judge_score": round(sum(all_judge)/n_in_domain, 2),
            "judge_model": "qwen3:8b",
            "generator_model": DEFAULT_MODEL,
        },
        "hallucination_refusal": {
            "num_ood_queries": n_ood,
            "correctly_refused": hallucination_checks["passed"],
            "refusal_rate": f"{100*hallucination_checks['passed']/n_ood:.1f}%" if n_ood > 0 else "N/A",
        },
        "response_time": {
            "num_total_queries": n_total,
            "mean_seconds": round(sum(all_times)/n_total, 1),
            "median_seconds": round(times_sorted[n_total//2], 1),
            "min_seconds": round(min(all_times), 1),
            "max_seconds": round(max(all_times), 1),
        },
    }

    out_path = os.path.join(DATA_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
