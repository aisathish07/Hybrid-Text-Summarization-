"""
Benchmark Testing Script for Hybrid Text Summarization System
Tests on CNN/DailyMail and XSum datasets
"""

import os
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Import our modules
from extractive.text_rank import extract_key_sentences
from clustering.semantic_cluster import SemanticClusterer
from abstractive.ensemble import AbstractiveEnsemble
from evaluation.metrics import evaluate_summary
from meta_selection.selector import select_best_summary

def test_dataset(dataset_name, samples, num_samples=10, language='en'):
    """
    Test the summarization system on a benchmark dataset.
    
    Args:
        dataset_name: Name of the dataset ('cnn_dailymail' or 'xsum')
        samples: Dataset samples (iterable)
        num_samples: Number of samples to test (default: 10)
        language: Language code (default: 'en')
    
    Returns:
        dict: Results with ROUGE scores and summaries
    """
    print(f"\n{'='*80}")
    print(f"Testing on {dataset_name.upper()} Dataset")
    print(f"Number of samples: {num_samples}")
    print(f"{'='*80}\n")
    
    # Initialize components
    clusterer = SemanticClusterer()
    ensemble = AbstractiveEnsemble()
    
    results = {
        'dataset': dataset_name,
        'num_samples': num_samples,
        'samples': [],
        'avg_rouge1': 0,
        'avg_rouge2': 0,
        'avg_rougeL': 0,
        'avg_bertscore': 0,
        'avg_coherence': 0
    }
    
    rouge1_total, rouge2_total, rougeL_total = 0, 0, 0
    bertscore_total, coherence_total = 0, 0
    
    # Process samples
    for i, sample in enumerate(tqdm(samples, desc="Processing samples", total=num_samples)):
        if i >= num_samples:
            break
        
        # Extract article and reference summary
        if dataset_name == 'cnn_dailymail':
            article = sample['article']
            reference = sample['highlights']
        elif dataset_name == 'xsum':
            article = sample['document']
            reference = sample['summary']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        try:
            # Run our hybrid pipeline
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            
            # 1. Extractive
            extractive_sents = extract_key_sentences(article, top_n=5, language=language)
            extractive_text = " ".join(extractive_sents)
            
            # 2. Clustering
            clustered_sents = clusterer.cluster_sentences(extractive_sents, n_clusters=3)
            clustered_text = " ".join(clustered_sents)
            
            # 3. Abstractive Ensemble
            candidates = ensemble.generate_candidates(clustered_text, max_length=150, min_length=30)
            
            # 4. Meta-Selection
            if candidates:
                best_model, best_summary, scores = select_best_summary(
                    candidates, 
                    clustered_text
                )
            else:
                best_summary = clustered_text
                scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'bertscore': 0, 'coherence': 0}
            
            # Evaluate against reference
            eval_scores = evaluate_summary(best_summary, reference)
            
            # Store results
            sample_result = {
                'id': i + 1,
                'article_length': len(article.split()),
                'reference_length': len(reference.split()),
                'summary_length': len(best_summary.split()),
                'extractive_length': len(extractive_text.split()),
                'scores': eval_scores,
                'summary': best_summary[:200] + "..." if len(best_summary) > 200 else best_summary
            }
            results['samples'].append(sample_result)
            
            # Accumulate scores
            rouge1_total += eval_scores.get('rouge1', 0)
            rouge2_total += eval_scores.get('rouge2', 0)
            rougeL_total += eval_scores.get('rougeL', 0)
            bertscore_total += eval_scores.get('bertscore', 0)
            coherence_total += eval_scores.get('coherence', 0)
            
            print(f"Summary (first 100 chars): {best_summary[:100]}...")
            print(f"ROUGE-1: {eval_scores.get('rouge1', 0):.4f}")
            print(f"ROUGE-2: {eval_scores.get('rouge2', 0):.4f}")
            print(f"ROUGE-L: {eval_scores.get('rougeL', 0):.4f}")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate averages
    num_processed = len(results['samples'])
    if num_processed > 0:
        results['avg_rouge1'] = rouge1_total / num_processed
        results['avg_rouge2'] = rouge2_total / num_processed
        results['avg_rougeL'] = rougeL_total / num_processed
        results['avg_bertscore'] = bertscore_total / num_processed
        results['avg_coherence'] = coherence_total / num_processed
    
    return results

def main():
    """Main testing function"""
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Testing for Hybrid Summarization')
    parser.add_argument('--dataset', type=str, choices=['cnn_dailymail', 'xsum', 'both'], 
                        default='both', help='Dataset to test')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of samples to test per dataset')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found. Installing...")
        os.system(f'"{sys.executable}" -m pip install datasets')
        from datasets import load_dataset
    
    all_results = {}
    
    # Test CNN/DailyMail
    if args.dataset in ['cnn_dailymail', 'both']:
        print("\n📰 Loading CNN/DailyMail dataset...")
        cnn_dataset = load_dataset("cnn_dailymail", "3.0.0", split='test')
        cnn_results = test_dataset('cnn_dailymail', cnn_dataset, args.num_samples)
        all_results['cnn_dailymail'] = cnn_results
        
        # Save results
        with open(os.path.join(args.output_dir, 'cnn_dailymail_results.json'), 'w') as f:
            json.dump(cnn_results, f, indent=2)
    
    # Test XSum
    if args.dataset in ['xsum', 'both']:
        print("\n📰 Loading XSum dataset...")
        xsum_dataset = load_dataset("xsum", split='test')
        xsum_results = test_dataset('xsum', xsum_dataset, args.num_samples)
        all_results['xsum'] = xsum_results
        
        # Save results
        with open(os.path.join(args.output_dir, 'xsum_results.json'), 'w') as f:
            json.dump(xsum_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK TEST RESULTS SUMMARY")
    print("="*80)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Samples Processed: {len(results['samples'])}/{results['num_samples']}")
        print(f"  Average ROUGE-1:   {results['avg_rouge1']:.4f}")
        print(f"  Average ROUGE-2:   {results['avg_rouge2']:.4f}")
        print(f"  Average ROUGE-L:   {results['avg_rougeL']:.4f}")
        print(f"  Average BERTScore: {results['avg_bertscore']:.4f}")
        print(f"  Average Coherence: {results['avg_coherence']:.4f}")
    
    print(f"\n✅ Results saved to: {args.output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
