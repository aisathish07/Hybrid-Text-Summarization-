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
print("\n[1/5] Downloading BERTScore model (roberta-large)...")
print("      This is ~1.4GB and may take a few minutes.")
try:
    from transformers import AutoModel, AutoTokenizer
    AutoModel.from_pretrained("roberta-large")
    AutoTokenizer.from_pretrained("roberta-large")
    print("      ✅ BERTScore model downloaded and cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

# 2. mBART (Multilingual BART)
print("\n[2/5] Downloading mBART model (mbart-large-50)...")
print("      This is ~2.4GB and may take several minutes.")
try:
    from transformers import AutoModelForSeq2SeqLM
    AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    print("      ✅ mBART model downloaded and cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

# 3. mT5 (Multilingual T5)
print("\n[3/5] Downloading mT5 model (mT5_multilingual_XLSum)...")
print("      This is ~1.2GB and may take a few minutes.")
try:
    AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    print("      ✅ mT5 model downloaded and cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

# 4. PEGASUS
print("\n[4/5] Downloading PEGASUS model (pegasus-cnn_dailymail)...")
print("      This is ~2.3GB and may take several minutes.")
try:
    AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
    AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    print("      ✅ PEGASUS model downloaded and cached!")
except Exception as e:
    print(f"      ❌ Failed: {e}")

# 5. Sentence-Transformers (for clustering + coherence)
print("\n[5/5] Downloading Sentence-Transformer models...")
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
