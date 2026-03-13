"""
Quick test script - tests 3 samples from CNN/DailyMail.
"""

from datasets import load_dataset

from abstractive.ensemble import AbstractiveEnsemble
from clustering.semantic_cluster import SemanticClusterer
from evaluation.metrics import Evaluator, evaluate_summary
from extractive.text_rank import extract_key_sentences
from meta_selection.selector import select_best_summary

print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)

print("\nInitializing components...")
clusterer = SemanticClusterer()
ensemble = AbstractiveEnsemble()
evaluator = Evaluator()

print("\nProcessing 3 samples...\n")

for i, sample in enumerate(dataset):
    if i >= 3:
        break

    article = sample["article"]
    reference = sample["highlights"]

    print("=" * 80)
    print(f"Sample {i + 1}")
    print("=" * 80)
    print(f"Article (first 200 chars): {article[:200]}...")
    print(f"\nReference Summary: {reference}")

    extractive = extract_key_sentences(article, top_n=5, language="en")
    clustered = clusterer.cluster_sentences(extractive, n_clusters=3)
    clustered_text = " ".join(clustered)

    candidates = ensemble.generate_candidates(clustered_text, max_length=150)

    if candidates:
        best_model, best_summary, scores = select_best_summary(candidates, clustered_text)
        generated_summary = best_summary
        print(f"\nGenerated Summary: {generated_summary}")
        print(f"\nBest Model: {best_model}")
        print(f"Scores: {scores}")
    else:
        best_model = "extractive_fallback"
        generated_summary = clustered_text
        print("\nNo candidates generated, using extractive summary")
        print(f"Summary: {generated_summary}")

    metrics = evaluate_summary(generated_summary, reference, evaluator=evaluator)
    print(f"Reference Metrics: {metrics}")
    print("\n")

print("Quick test complete!")
