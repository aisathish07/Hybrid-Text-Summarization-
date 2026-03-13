"""
Benchmark runner that saves human-readable results to a text report.
"""

from datetime import datetime
import argparse

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


def run_benchmark(dataset_name, num_samples=10, output_file=None):
    """
    Run the benchmark on a dataset and save the results to a text report.
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{dataset_name}_{num_samples}samples_{timestamp}.txt"

    print("\n" + "=" * 80)
    print(f"BENCHMARK: {dataset_name.upper()}")
    print(f"Samples: {num_samples}")
    print(f"Output: {output_file}")
    print("=" * 80 + "\n")

    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset_split(dataset_name)

    print("Initializing models...")
    clusterer = SemanticClusterer()
    ensemble = AbstractiveEnsemble()
    evaluator = Evaluator()

    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("HYBRID TEXT SUMMARIZATION - BENCHMARK RESULTS\n")
        handle.write(f"Dataset: {dataset_name.upper()}\n")
        handle.write(f"Samples Requested: {num_samples}\n")
        handle.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write("=" * 80 + "\n\n")

        total_rouge1 = total_rouge2 = total_rougeL = 0.0
        total_bertscore = total_coherence = 0.0
        successful_samples = 0

        for i, sample in enumerate(tqdm(dataset, desc="Processing", total=num_samples), start=1):
            if i > num_samples:
                break

            try:
                if dataset_name == "cnn_dailymail":
                    article = sample["article"]
                    reference = sample["highlights"]
                else:
                    article = sample["document"]
                    reference = sample["summary"]

                print(f"\nProcessing sample {i}/{num_samples}...")

                extractive_sents = extract_key_sentences(article, top_n=5, language="en")
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

                total_rouge1 += eval_scores.get("rouge1", 0.0)
                total_rouge2 += eval_scores.get("rouge2", 0.0)
                total_rougeL += eval_scores.get("rougeL", 0.0)
                total_bertscore += eval_scores.get("bertscore", 0.0)
                total_coherence += eval_scores.get("coherence", 0.0)
                successful_samples += 1

                handle.write("\n" + "=" * 80 + "\n")
                handle.write(f"SAMPLE {i}\n")
                handle.write("=" * 80 + "\n\n")
                handle.write("ORIGINAL ARTICLE:\n")
                handle.write(f"{article[:500]}{'...' if len(article) > 500 else ''}\n\n")
                handle.write("REFERENCE SUMMARY:\n")
                handle.write(f"{reference}\n\n")
                handle.write(f"GENERATED SUMMARY (Model: {best_model}):\n")
                handle.write(f"{best_summary}\n\n")
                handle.write("ACCURACY SCORES:\n")
                handle.write(f"  ROUGE-1:   {eval_scores.get('rouge1', 0.0):.4f}\n")
                handle.write(f"  ROUGE-2:   {eval_scores.get('rouge2', 0.0):.4f}\n")
                handle.write(f"  ROUGE-L:   {eval_scores.get('rougeL', 0.0):.4f}\n")
                handle.write(f"  BERTScore: {eval_scores.get('bertscore', 0.0):.4f}\n")
                handle.write(f"  Coherence: {eval_scores.get('coherence', 0.0):.4f}\n")

                print(f"OK sample {i} completed - ROUGE-L: {eval_scores.get('rougeL', 0.0):.4f}")
            except Exception as exc:
                print(f"ERROR processing sample {i}: {exc}")
                handle.write("\n" + "=" * 80 + "\n")
                handle.write(f"SAMPLE {i} - ERROR\n")
                handle.write("=" * 80 + "\n")
                handle.write(f"Error: {exc}\n")

        handle.write("\n" + "=" * 80 + "\n")
        handle.write("OVERALL STATISTICS\n")
        handle.write("=" * 80 + "\n\n")

        if successful_samples:
            handle.write(f"Samples Processed: {successful_samples}/{num_samples}\n")
            handle.write(f"Average ROUGE-1:   {total_rouge1 / successful_samples:.4f}\n")
            handle.write(f"Average ROUGE-2:   {total_rouge2 / successful_samples:.4f}\n")
            handle.write(f"Average ROUGE-L:   {total_rougeL / successful_samples:.4f}\n")
            handle.write(f"Average BERTScore: {total_bertscore / successful_samples:.4f}\n")
            handle.write(f"Average Coherence: {total_coherence / successful_samples:.4f}\n")
        else:
            handle.write("No samples were successfully processed.\n")

        handle.write("\n" + "=" * 80 + "\n")
        handle.write("END OF REPORT\n")
        handle.write("=" * 80 + "\n")

    print("\nBenchmark complete.")
    print(f"Results saved to: {output_file}")
    if successful_samples:
        print("\nSummary:")
        print(f"  Processed: {successful_samples}/{num_samples} samples")
        print(f"  Avg ROUGE-L: {total_rougeL / successful_samples:.4f}")
        print(f"  Avg BERTScore: {total_bertscore / successful_samples:.4f}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark on large datasets and save results to a text file."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cnn_dailymail", "xsum"],
        required=True,
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text file path. A timestamped name is used by default.",
    )

    args = parser.parse_args()
    run_benchmark(args.dataset, args.samples, args.output)


if __name__ == "__main__":
    main()
