"""One-command setup: download data, build indexes, and verify everything works."""

import subprocess
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DIR = os.path.dirname(os.path.abspath(__file__))


def run(script):
    print(f"\n{'='*60}\nRunning {script}...\n{'='*60}")
    result = subprocess.run([sys.executable, os.path.join(DIR, script)], cwd=DIR)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    print("Academic Research Assistant - Setup")
    print("This will download ~4GB of data and build indexes (~1 hour total).\n")

    # Step 1: Install dependencies
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(DIR, "requirements.txt"), "-q"])

    # Step 2: Download arXiv data
    run("download_data.py")

    # Step 3: Ingest into ChromaDB + BM25
    run("ingest.py")

    # Step 4: Add PubMed (optional but recommended)
    run("add_pubmed.py")

    print(f"\n{'='*60}")
    print("Setup complete! Run the app with:")
    print("  KMP_DUPLICATE_LIB_OK=TRUE streamlit run app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
