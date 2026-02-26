"""
Benchmark Runner - Processes large datasets and saves summaries to text files
Outputs: Human-readable text files with summaries and accuracy scores
"""

import os
import sys
from datetime import datetime
from tqdm import tqdm

# Import our modules
from extractive.text_rank import extract_key_sentences
from clustering.semantic_cluster import SemanticClusterer
from abstractive.ensemble import AbstractiveEnsemble
from evaluation.metrics import evaluate_summary
from meta_selection.selector import select_best_summary

def run_benchmark(dataset_name, num_samples=10, output_file=None):
    """
    Run benchmark on dataset and save results to text file.
    
    Args:
        dataset_name: 'cnn_dailymail' or 'xsum'
        num_samples: Number of samples to process
        output_file: Output text file path (auto-generated if None)
    """
    
    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system(f'"{sys.executable}" -m pip install datasets')
        from datasets import load_dataset
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{dataset_name}_{num_samples}samples_{timestamp}.txt"
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {dataset_name.upper()}")
    print(f"Samples: {num_samples}")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'cnn_dailymail':
        dataset = load_dataset("cnn_dailymail", "3.0.0", split='test')
    elif dataset_name == 'xsum':
        dataset = load_dataset("xsum", split='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Initialize components
    print("Initializing models...")
    clusterer = SemanticClusterer()
    ensemble = AbstractiveEnsemble()
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("="*80 + "\n")
        f.write(f"HYBRID TEXT SUMMARIZATION - BENCHMARK RESULTS\n")
        f.write(f"Dataset: {dataset_name.upper()}\n")
        f.write(f"Samples Processed: {num_samples}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Track averages
        total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
        total_bertscore, total_coherence = 0, 0
        successful_samples = 0
        
        # Process samples
        for i, sample in enumerate(tqdm(dataset, desc="Processing", total=num_samples)):
            if i >= num_samples:
                break
            
            try:
                # Extract text and reference
                if dataset_name == 'cnn_dailymail':
                    article = sample['article']
                    reference = sample['highlights']
                elif dataset_name == 'xsum':
                    article = sample['document']
                    reference = sample['summary']
                
                # Run pipeline
                print(f"\nProcessing Sample {i+1}/{num_samples}...")
                
                # 1. Extractive
                extractive_sents = extract_key_sentences(article, top_n=5, language='en')
                extractive_text = " ".join(extractive_sents)
                
                # 2. Clustering
                clustered_sents = clusterer.cluster_sentences(extractive_sents, n_clusters=3)
                clustered_text = " ".join(clustered_sents)
                
                # 3. Abstractive Ensemble
                candidates = ensemble.generate_candidates(clustered_text, max_length=150, min_length=30)
                
                # 4. Meta-Selection
                if candidates:
                    best_model, best_summary, scores = select_best_summary(candidates, clustered_text)
                else:
                    best_summary = clustered_text
                    best_model = "extractive_fallback"
                    scores = {}
                
                # 5. Evaluate against reference
                eval_scores = evaluate_summary(best_summary, reference)
                
                # Accumulate scores
                total_rouge1 += eval_scores.get('rouge1', 0)
                total_rouge2 += eval_scores.get('rouge2', 0)
                total_rougeL += eval_scores.get('rougeL', 0)
                total_bertscore += eval_scores.get('bertscore', 0)
                total_coherence += eval_scores.get('coherence', 0)
                successful_samples += 1
                
                # Write to file
                f.write(f"\n{'='*80}\n")
                f.write(f"SAMPLE {i+1}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"ORIGINAL ARTICLE:\n")
                f.write(f"{article[:500]}{'...' if len(article) > 500 else ''}\n\n")
                
                f.write(f"REFERENCE SUMMARY:\n")
                f.write(f"{reference}\n\n")
                
                f.write(f"GENERATED SUMMARY (Model: {best_model}):\n")
                f.write(f"{best_summary}\n\n")
                
                f.write(f"ACCURACY SCORES:\n")
                f.write(f"  ROUGE-1:   {eval_scores.get('rouge1', 0):.4f}\n")
                f.write(f"  ROUGE-2:   {eval_scores.get('rouge2', 0):.4f}\n")
                f.write(f"  ROUGE-L:   {eval_scores.get('rougeL', 0):.4f}\n")
                f.write(f"  BERTScore: {eval_scores.get('bertscore', 0):.4f}\n")
                f.write(f"  Coherence: {eval_scores.get('coherence', 0):.4f}\n")
                f.write(f"\n")
                
                print(f"✓ Sample {i+1} completed - ROUGE-L: {eval_scores.get('rougeL', 0):.4f}")
                
            except Exception as e:
                print(f"✗ Error processing sample {i+1}: {e}")
                f.write(f"\n{'='*80}\n")
                f.write(f"SAMPLE {i+1} - ERROR\n")
                f.write(f"{'='*80}\n")
                f.write(f"Error: {str(e)}\n\n")
                continue
        
        # Write summary statistics
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        if successful_samples > 0:
            f.write(f"Samples Processed: {successful_samples}/{num_samples}\n")
            f.write(f"Average ROUGE-1:   {total_rouge1/successful_samples:.4f}\n")
            f.write(f"Average ROUGE-2:   {total_rouge2/successful_samples:.4f}\n")
            f.write(f"Average ROUGE-L:   {total_rougeL/successful_samples:.4f}\n")
            f.write(f"Average BERTScore: {total_bertscore/successful_samples:.4f}\n")
            f.write(f"Average Coherence: {total_coherence/successful_samples:.4f}\n")
        else:
            f.write("No samples were successfully processed.\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✅ Benchmark complete!")
    print(f"📄 Results saved to: {output_file}")
    print(f"\nSummary:")
    if successful_samples > 0:
        print(f"  Processed: {successful_samples}/{num_samples} samples")
        print(f"  Avg ROUGE-L: {total_rougeL/successful_samples:.4f}")
        print(f"  Avg BERTScore: {total_bertscore/successful_samples:.4f}")
    
    return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run benchmark on large datasets and save results to text file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10 samples from CNN/DailyMail
  python run_benchmark.py --dataset cnn_dailymail --samples 10
  
  # Process 50 samples from XSum
  python run_benchmark.py --dataset xsum --samples 50
  
  # Process 100 samples with custom output file
  python run_benchmark.py --dataset cnn_dailymail --samples 100 --output my_results.txt
        """
    )
    
    parser.add_argument('--dataset', type=str, choices=['cnn_dailymail', 'xsum'],
                        required=True, help='Dataset to benchmark')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to process (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output text file (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    run_benchmark(args.dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
