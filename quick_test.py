"""
Quick test script - Tests 3 samples from CNN/DailyMail
"""

from datasets import load_dataset
from extractive.text_rank import extract_key_sentences
from clustering.semantic_cluster import SemanticClusterer
from abstractive.ensemble import AbstractiveEnsemble
from meta_selection.selector import select_best_summary

print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split='test', streaming=True)

print("\nInitializing components...")
clusterer = SemanticClusterer()
ensemble = AbstractiveEnsemble()

print("\nProcessing 3 samples...\n")

for i, sample in enumerate(dataset):
    if i >= 3:
        break
    
    article = sample['article']
    reference = sample['highlights']
    
    print(f"{'='*80}")
    print(f"Sample {i+1}")
    print(f"{'='*80}")
    print(f"Article (first 200 chars): {article[:200]}...")
    print(f"\nReference Summary: {reference}")
    
    # Pipeline
    extractive = extract_key_sentences(article, top_n=5, language='en')
    clustered = clusterer.cluster_sentences(extractive, n_clusters=3)
    clustered_text = " ".join(clustered)
    
    candidates = ensemble.generate_candidates(clustered_text, max_length=150)
    
    if candidates:
        best_model, best_summary, scores = select_best_summary(candidates, clustered_text)
        print(f"\nGenerated Summary: {best_summary}")
        print(f"\nBest Model: {best_model}")
        print(f"Scores: {scores}")
    else:
        print("\nNo candidates generated, using extractive summary")
        print(f"Summary: {clustered_text}")
    
    print("\n")

print("✅ Quick test complete!")
