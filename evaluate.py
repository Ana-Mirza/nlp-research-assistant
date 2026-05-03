"""Evaluate the RAG system with custom metrics and LLM-as-Judge."""

import json
import time
import re
import os
from rag import research_assistant
from llm import generate
from config import DEFAULT_MODEL, DATA_DIR

TEST_QUERIES = [
    {"query": "federated neural topic models", "keywords": ["federated", "topic model"]},
    {"query": "transformer architectures for protein folding", "keywords": ["transformer", "protein"]},
    {"query": "reinforcement learning for robotics", "keywords": ["reinforcement", "robot"]},
    {"query": "attention mechanisms in computer vision", "keywords": ["attention", "vision"]},
    {"query": "graph neural networks for drug discovery", "keywords": ["graph", "drug"]},
    {"query": "few-shot learning for NLP", "keywords": ["few-shot", "language"]},
    {"query": "self-supervised learning for speech recognition", "keywords": ["self-supervised", "speech"]},
    {"query": "neural architecture search", "keywords": ["architecture search", "NAS"]},
    {"query": "knowledge distillation in large language models", "keywords": ["distillation", "language model"]},
    {"query": "adversarial robustness of deep learning models", "keywords": ["adversarial", "robust"]},
    {"query": "multi-modal learning combining vision and language", "keywords": ["multi-modal", "vision", "language"]},
    {"query": "continual learning without catastrophic forgetting", "keywords": ["continual", "forgetting"]},
    {"query": "efficient inference for large language models", "keywords": ["efficient", "inference", "language model"]},
    {"query": "zero-shot cross-lingual transfer", "keywords": ["cross-lingual", "zero-shot"]},
    {"query": "diffusion models for text generation", "keywords": ["diffusion", "text"]},
]


def check_relevance(papers, keywords):
    """Check if at least one paper is relevant based on keyword matching."""
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def llm_judge_score(query, answer, papers):
    """Use LLM-as-Judge to rate answer quality (1-5). Score first, explanation separate."""
    context = "\n".join(f'- "{p["title"]}"' for p in papers[:5])
    prompt = (
        f"Rate the following answer on a scale of 1-5 for quality as a research assistant response.\n"
        f"Consider: (1) relevance to the query, (2) grounding in the retrieved papers, (3) identification of key differences.\n\n"
        f"Query: {query}\n"
        f"Retrieved papers:\n{context}\n\n"
        f"Answer:\n{answer[:1000]}\n\n"
        f"Reply with ONLY a single number from 1 to 5."
    )
    resp = generate(prompt, model=DEFAULT_MODEL, temperature=0.0)
    # Extract first digit
    match = re.search(r'[1-5]', resp)
    return int(match.group()) if match else 3


def main():
    print(f"Evaluating RAG system with {len(TEST_QUERIES)} queries...")
    print("=" * 70)

    results = []
    times = []
    scores = []
    coverage_hits = 0
    judge_scores = []

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        keywords = test["keywords"]
        print(f"\n[{i+1}/{len(TEST_QUERIES)}] {query}")

        t0 = time.time()
        result = research_assistant(query, top_k=5)
        elapsed = time.time() - t0
        times.append(elapsed)

        papers = result.get("papers", [])
        answer = result.get("answer", "")
        avg_score = sum(p["score"] for p in papers) / len(papers) if papers else 0

        # Document coverage
        relevant = check_relevance(papers, keywords)
        if relevant:
            coverage_hits += 1

        # Retrieval quality
        scores.append(avg_score)

        # LLM-as-Judge
        judge = llm_judge_score(query, answer, papers)
        judge_scores.append(judge)

        print(f"  Papers: {len(papers)} | Relevant: {relevant} | Avg score: {avg_score:.3f} | Judge: {judge}/5 | Time: {elapsed:.1f}s")

        results.append({
            "query": query,
            "num_papers": len(papers),
            "relevant": relevant,
            "avg_reranker_score": round(avg_score, 3),
            "judge_score": judge,
            "time_seconds": round(elapsed, 2),
            "paper_titles": [p["title"] for p in papers],
        })

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    n = len(TEST_QUERIES)
    print(f"Document Coverage:     {coverage_hits}/{n} ({100*coverage_hits/n:.1f}%)")
    print(f"Avg Retrieval Score:   {sum(scores)/n:.3f}")
    print(f"Avg LLM Judge Score:   {sum(judge_scores)/n:.2f}/5")
    print(f"Response Time (mean):  {sum(times)/n:.1f}s")
    print(f"Response Time (min):   {min(times):.1f}s")
    print(f"Response Time (max):   {max(times):.1f}s")
    print(f"Response Time (median):{sorted(times)[n//2]:.1f}s")

    # Save results
    output = {
        "summary": {
            "document_coverage": f"{coverage_hits}/{n} ({100*coverage_hits/n:.1f}%)",
            "avg_retrieval_score": round(sum(scores)/n, 3),
            "avg_judge_score": round(sum(judge_scores)/n, 2),
            "mean_time_seconds": round(sum(times)/n, 1),
            "min_time_seconds": round(min(times), 1),
            "max_time_seconds": round(max(times), 1),
        },
        "per_query": results,
    }
    out_path = os.path.join(DATA_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
