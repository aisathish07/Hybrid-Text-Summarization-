"""
Benchmark summary script that saves structured JSON results.
"""

from datetime import datetime
import argparse
import json
import os

from tqdm import tqdm

from abstractive.ensemble import AbstractiveEnsemble
from clustering.semantic_cluster import SemanticClusterer
from evaluation.metrics import Evaluator, evaluate_summary
from extractive.text_rank import extract_key_sentences
from meta_selection.selector import select_best_summary


def load_dataset_split(dataset_name):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install packages with `pip install -r requirements.txt`.") from exc

    if dataset_name == "cnn_dailymail":
        return load_dataset("cnn_dailymail", "3.0.0", split="test")
    if dataset_name == "xsum":
        return load_dataset("xsum", split="test")
    raise ValueError(f"Unknown dataset: {dataset_name}")


def benchmark_dataset(dataset_name, samples, num_samples=10, language="en"):
    """
    Collect JSON-friendly benchmark results for a dataset split.
    """
    print("\n" + "=" * 80)
    print(f"Testing on {dataset_name.upper()} dataset")
    print(f"Number of samples: {num_samples}")
    print("=" * 80 + "\n")

    clusterer = SemanticClusterer()
    ensemble = AbstractiveEnsemble()
    evaluator = Evaluator()

    results = {
        "dataset": dataset_name,
        "num_samples": num_samples,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "samples": [],
        "avg_rouge1": 0.0,
        "avg_rouge2": 0.0,
        "avg_rougeL": 0.0,
        "avg_bertscore": 0.0,
        "avg_coherence": 0.0,
    }

    rouge1_total = rouge2_total = rougeL_total = 0.0
    bertscore_total = coherence_total = 0.0

    for i, sample in enumerate(tqdm(samples, desc="Processing samples", total=num_samples), start=1):
        if i > num_samples:
            break

        if dataset_name == "cnn_dailymail":
            article = sample["article"]
            reference = sample["highlights"]
        else:
            article = sample["document"]
            reference = sample["summary"]

        try:
            print(f"\n--- Sample {i}/{num_samples} ---")

            extractive_sents = extract_key_sentences(article, top_n=5, language=language)
            extractive_text = " ".join(extractive_sents)

            clustered_sents = clusterer.cluster_sentences(extractive_sents, n_clusters=3)
            clustered_text = " ".join(clustered_sents)

            candidates = ensemble.generate_candidates(clustered_text, max_length=150, min_length=30)

            if candidates:
                best_model, best_summary, _ = select_best_summary(candidates, clustered_text)
            else:
                best_model = "extractive_fallback"
                best_summary = clustered_text

            eval_scores = evaluate_summary(best_summary, reference, evaluator=evaluator)

            results["samples"].append(
                {
                    "id": i,
                    "best_model": best_model,
                    "article_length": len(article.split()),
                    "reference_length": len(reference.split()),
                    "summary_length": len(best_summary.split()),
                    "extractive_length": len(extractive_text.split()),
                    "scores": eval_scores,
                    "summary": best_summary[:200] + "..." if len(best_summary) > 200 else best_summary,
                }
            )

            rouge1_total += eval_scores.get("rouge1", 0.0)
            rouge2_total += eval_scores.get("rouge2", 0.0)
            rougeL_total += eval_scores.get("rougeL", 0.0)
            bertscore_total += eval_scores.get("bertscore", 0.0)
            coherence_total += eval_scores.get("coherence", 0.0)

            print(f"Summary (first 100 chars): {best_summary[:100]}...")
            print(f"ROUGE-1: {eval_scores.get('rouge1', 0.0):.4f}")
            print(f"ROUGE-2: {eval_scores.get('rouge2', 0.0):.4f}")
            print(f"ROUGE-L: {eval_scores.get('rougeL', 0.0):.4f}")
        except Exception as exc:
            print(f"Error processing sample {i}: {exc}")
            continue

    num_processed = len(results["samples"])
    if num_processed:
        results["avg_rouge1"] = rouge1_total / num_processed
        results["avg_rouge2"] = rouge2_total / num_processed
        results["avg_rougeL"] = rougeL_total / num_processed
        results["avg_bertscore"] = bertscore_total / num_processed
        results["avg_coherence"] = coherence_total / num_processed

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark the summarization system and save JSON results.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cnn_dailymail", "xsum", "both"],
        default="both",
        help="Dataset to test.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test per dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where JSON results are written.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    if args.dataset in ["cnn_dailymail", "both"]:
        print("\nLoading CNN/DailyMail dataset...")
        cnn_dataset = load_dataset_split("cnn_dailymail")
        cnn_results = benchmark_dataset("cnn_dailymail", cnn_dataset, args.num_samples)
        all_results["cnn_dailymail"] = cnn_results
        with open(os.path.join(args.output_dir, "cnn_dailymail_results.json"), "w", encoding="utf-8") as handle:
            json.dump(cnn_results, handle, indent=2)

    if args.dataset in ["xsum", "both"]:
        print("\nLoading XSum dataset...")
        xsum_dataset = load_dataset_split("xsum")
        xsum_results = benchmark_dataset("xsum", xsum_dataset, args.num_samples)
        all_results["xsum"] = xsum_results
        with open(os.path.join(args.output_dir, "xsum_results.json"), "w", encoding="utf-8") as handle:
            json.dump(xsum_results, handle, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK TEST RESULTS SUMMARY")
    print("=" * 80)

    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Samples Processed: {len(results['samples'])}/{results['num_samples']}")
        print(f"  Average ROUGE-1:   {results['avg_rouge1']:.4f}")
        print(f"  Average ROUGE-2:   {results['avg_rouge2']:.4f}")
        print(f"  Average ROUGE-L:   {results['avg_rougeL']:.4f}")
        print(f"  Average BERTScore: {results['avg_bertscore']:.4f}")
        print(f"  Average Coherence: {results['avg_coherence']:.4f}")

    print(f"\nResults saved to: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
