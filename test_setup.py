import sys

print("Testing imports...")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    import sentence_transformers
    print(f"Sentence-Transformers version: {sentence_transformers.__version__}")
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    import nltk
    print(f"NLTK version: {nltk.__version__}")
    import spacy
    print(f"Spacy version: {spacy.__version__}")
    # note: rouge_score is installed as rouge-score but imported as rouge_score usually, checking...
    import rouge_score
    print("rouge_score imported")
    import bert_score
    print(f"BERTScore version: {bert_score.__version__}")
    import datasets
    print(f"Datasets version: {datasets.__version__}")
    import requests
    print(f"Requests version: {requests.__version__}")
    import fastapi
    print(f"FastAPI version: {fastapi.__version__}")
    import uvicorn
    print(f"Uvicorn version: {uvicorn.__version__}")
    import pydantic
    print(f"Pydantic version: {pydantic.__version__}")

    print("\nAll imports successful!")

except ImportError as e:
    print(f"\nImport failed: {e}")
    sys.exit(1)
