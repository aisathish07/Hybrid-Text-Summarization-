"""
Pre-download all required HuggingFace models.
Run this ONCE before starting the Streamlit app to avoid runtime download timeouts.

Usage:
    python download_models.py
"""

import sys

print("=" * 60)
print("  Model Pre-Downloader for Hybrid Summarizer")
print("=" * 60)

# 1. BERTScore model (roberta-large ~1.4GB)
print("\n[1/3] Downloading BERTScore model (roberta-large)...")
print("      This is ~1.4GB and may take a few minutes.")
try:
    from transformers import AutoModel, AutoTokenizer
    AutoModel.from_pretrained("roberta-large")
    AutoTokenizer.from_pretrained("roberta-large")
    print("      ✅ BERTScore model downloaded and cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

# 2. Abstractive model (mBART)
print("\n[2/3] Downloading Abstractive model (mBART-large-50)...")
print("      This is ~2.4GB and may take several minutes.")
try:
    from transformers import AutoModelForSeq2SeqLM
    AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    print("      ✅ mBART model downloaded and cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

# 3. Sentence-Transformers (for clustering + coherence)
print("\n[3/3] Downloading Sentence-Transformer models...")
try:
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("      ✅ Multilingual clustering model cached!")
    SentenceTransformer("all-MiniLM-L6-v2")
    print("      ✅ Coherence scoring model cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

print("\n" + "=" * 60)
print("  All downloads complete! You can now run:")
print("  streamlit run app.py")
print("=" * 60)
