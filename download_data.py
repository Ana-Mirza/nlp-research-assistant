"""Download and filter arXiv metadata from HuggingFace."""

import os
from datasets import load_dataset
from config import ARXIV_CATEGORIES, MIN_YEAR, DATA_DIR

MAX_PAPERS = None  # No cap — use all filtered papers


def matches_categories(categories: str) -> bool:
    return any(cat in categories for cat in ARXIV_CATEGORIES)


def parse_year(update_date) -> int:
    try:
        if hasattr(update_date, "year"):
            return update_date.year
        return int(str(update_date).split("-")[0])
    except (ValueError, TypeError, AttributeError):
        return 0


def main():
    output_path = os.path.join(DATA_DIR, "arxiv_filtered.parquet")

    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}, skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading arXiv metadata from HuggingFace (full parquet, ~3GB)...")
    print("This is a one-time bulk download — much faster than streaming.")
    dataset = load_dataset(
        "librarian-bots/arxiv-metadata-snapshot", split="train"
    )
    print(f"Downloaded {len(dataset):,} papers. Filtering in parallel...")

    filtered = dataset.filter(
        lambda x: matches_categories(x["categories"]) and parse_year(x["update_date"]) >= MIN_YEAR,
        num_proc=4,
    )
    print(f"Filtered to {len(filtered):,} papers (categories={ARXIV_CATEGORIES}, year>={MIN_YEAR}).")

    if MAX_PAPERS and len(filtered) > MAX_PAPERS:
        # Shuffle and cap to keep indexing fast
        filtered = filtered.shuffle(seed=42).select(range(MAX_PAPERS))
        print(f"Capped to {MAX_PAPERS:,} papers for practical indexing speed.")

    filtered.to_parquet(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
