
import argparse
import sys
import os
from functools import lru_cache

# Ensure the current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def _load_pipeline_components():
    """
    Import heavy pipeline components lazily so UI startup does not pay the
    transformer and embedding import cost before the user requests a summary.
    """
    from extractive.text_rank import extract_key_sentences
    from clustering.semantic_cluster import SemanticClusterer
    from abstractive.ensemble import AbstractiveEnsemble
    from evaluation.metrics import Evaluator
    from meta_selection.selector import MetaSelector

    return (
        extract_key_sentences,
        SemanticClusterer,
        AbstractiveEnsemble,
        Evaluator,
        MetaSelector,
    )


@lru_cache(maxsize=1)
def get_clusterer():
    _, SemanticClusterer, _, _, _ = _load_pipeline_components()
    return SemanticClusterer()


@lru_cache(maxsize=1)
def get_ensemble():
    _, _, AbstractiveEnsemble, _, _ = _load_pipeline_components()
    return AbstractiveEnsemble()


@lru_cache(maxsize=1)
def get_evaluator():
    _, _, _, Evaluator, _ = _load_pipeline_components()
    return Evaluator(coverage_model=get_clusterer().model)


def warm_runtime(load_models=False):
    """
    Prime the heavy runtime objects so a remote backend can pay startup cost
    once instead of on every request.
    """
    get_clusterer()
    evaluator = get_evaluator()

    if getattr(evaluator, "preload_bertscore", None):
        evaluator.preload_bertscore()

    if load_models:
        ensemble = get_ensemble()
        for model_name in ensemble.get_available_models():
            ensemble.load_model(model_name)

def run_summarization_pipeline(text, top_n=5, clusters=3, max_length=150, language='en', progress_callback=None):
    """
    Runs the full hybrid summarization pipeline on the provided text.
    Returns a dictionary with all intermediate and final outputs.

    Args:
        text: Input text to summarize
        top_n: Number of sentences for extractive summary
        clusters: Number of semantic clusters to select
        max_length: Max length for abstractive summary
        language: Language code (e.g. en, gu, hi)
        progress_callback: Optional callback function(stage_name, progress_percent) for progress updates
    """
    # Helper to call progress callback
    def report_progress(stage_name, percent):
        if progress_callback:
            progress_callback(stage_name, percent)

    (
        extract_key_sentences,
        _,
        _,
        _,
        MetaSelector,
    ) = _load_pipeline_components()

    # 1. Extractive Summary
    report_progress("Extracting key sentences with TextRank", 10)
    extractive_summary_list = extract_key_sentences(text, top_n=top_n, language=language)
    extractive_summary_text = " ".join(extractive_summary_list)
    report_progress("Text extraction complete", 25)

    # 2. Semantic Clustering
    report_progress("Performing semantic clustering", 30)
    clusterer = get_clusterer()
    clustered_sentences = clusterer.cluster_and_select(extractive_summary_list, n_clusters=clusters)
    clustered_summary_text = " ".join(clustered_sentences)
    report_progress("Clustering complete", 50)

    # 3. Abstractive Ensemble
    report_progress("Generating abstractive summaries", 55)
    ensemble = get_ensemble()
    input_to_abstractive = clustered_summary_text

    # Generate candidates one by one for better progress tracking
    candidates = {}
    try:
        candidate_models = ensemble.get_available_models()
    except AttributeError:
        # If method doesn't exist, use default models
        candidate_models = ["t5", "bart", "pegasus"]

    for i, model_name in enumerate(candidate_models):
        report_progress(f"Running {model_name}...", 55 + (i * 15 // max(len(candidate_models), 1)))
        try:
            candidate = ensemble.generate_single_candidate(input_to_abstractive, model_name, max_length, language=language)
            candidates[model_name] = candidate
        except Exception as e:
            print(f"Warning: {model_name} failed: {e}")
            candidates[model_name] = ""

    # Fallback if no candidates generated
    if not candidates:
        candidates = ensemble.generate_candidates(input_to_abstractive, max_length=max_length, language=language)

    report_progress("Abstractive generation complete", 85)

    # 4. Meta-Selection
    report_progress("Evaluating and selecting best summary", 90)
    evaluator = get_evaluator()
    selector = MetaSelector(evaluator)

    # Evaluate candidates against the clustered (optimized extractive) summary
    best_model, best_summary, scores = selector.select_best_summary(
        candidates,
        clustered_summary_text,
        language=language,
    )

    english_translation = None
    if best_summary and language != 'en':
        report_progress("Translating summary to English", 96)
        try:
            english_translation = ensemble.translate_to_english(best_summary, source_language=language)
        except Exception as e:
            print(f"Warning: translation failed: {e}")
            english_translation = None

    report_progress("Summary selection complete", 100)

    return {
        "extractive_list": extractive_summary_list,
        "extractive_text": extractive_summary_text,
        "clustered_list": clustered_sentences,
        "clustered_text": clustered_summary_text,
        "candidates": candidates,
        "best_model": best_model,
        "best_summary": best_summary,
        "scores": scores,
        "english_translation": english_translation,
    }


# ---- Individual Pipeline Steps for Real-time Progress ----

def step1_extractive(text, top_n=5, language='en'):
    """Stage 1: Extract key sentences using TextRank"""
    extract_key_sentences, _, _, _, _ = _load_pipeline_components()
    extractive_summary_list = extract_key_sentences(text, top_n=top_n, language=language)
    return {
        "extractive_list": extractive_summary_list,
        "extractive_text": " ".join(extractive_summary_list)
    }


def step2_clustering(extractive_list, n_clusters=3):
    """Stage 2: Semantic Clustering"""
    clusterer = get_clusterer()
    clustered_sentences = clusterer.cluster_and_select(extractive_list, n_clusters=n_clusters)
    return {
        "clustered_list": clustered_sentences,
        "clustered_text": " ".join(clustered_sentences)
    }


def step3_abstractive(clustered_text, max_length=150):
    """Stage 3: Generate abstractive candidates"""
    ensemble = get_ensemble()
    candidates = ensemble.generate_candidates(clustered_text, max_length=max_length)
    return {"candidates": candidates}


def step4_selection(clustered_text, candidates, language='en'):
    """Stage 4: Meta-selection of best summary"""
    _, _, _, _, MetaSelector = _load_pipeline_components()
    evaluator = get_evaluator()
    selector = MetaSelector(evaluator)
    best_model, best_summary, scores = selector.select_best_summary(
        candidates,
        clustered_text,
        language=language,
    )
    return {
        "best_model": best_model,
        "best_summary": best_summary,
        "scores": scores
    }

def main():
    parser = argparse.ArgumentParser(description="Hybrid Text Summarization System")
    parser.add_argument("--input_file", type=str, help="Path to input text file")
    parser.add_argument("--text", type=str, help="Input text string (optional, if file not provided)")
    parser.add_argument("--top_n", type=int, default=5, help="Number of sentences for extractive summary")
    parser.add_argument("--clusters", type=int, default=3, help="Number of semantic clusters to select")
    parser.add_argument("--max_length", type=int, default=150, help="Max length for abstractive summary")
    parser.add_argument("--language", type=str, default='en', help="Language code (e.g. en, gu, hi)")
    
    args = parser.parse_args()
    
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif args.text:
        text = args.text
    else:
        print("Please provide --input_file or --text")
        return

    print(f"\n=== Hybrid Text Summarization System ({args.language}) ===\n")
    
    results = run_summarization_pipeline(
        text=text,
        top_n=args.top_n,
        clusters=args.clusters,
        max_length=args.max_length,
        language=args.language
    )
    
    print("Step 1: Extractive Summarization (TextRank)...")
    print(f"Extractive Summary: {results['extractive_text'][:100]}...")
    
    print("\nStep 2: Semantic Clustering...")
    print(f"Clustered Summary: {results['clustered_text'][:100]}...")
    
    print("\nStep 3: Abstractive Ensemble Generation...")
    print(f"Input to Abstractive Models (Length: {len(results['clustered_text'])} chars)")
    
    print("\nCandidates Generated:")
    for model, summary in results['candidates'].items():
        print(f"[{model}]: {summary[:100]}...")
        
    print("\nStep 4: Meta-Selection...")
    print("\n=== FINAL RESULTS ===")
    print(f"Best Model: {results['best_model']}")
    print(f"Selected Summary:\n{results['best_summary']}")
    if results.get('english_translation'):
        print(f"\nEnglish Translation:\n{results['english_translation']}")
    
    print("\nScores:")
    for m, s in results['scores'].items():
        print(f"[{m}] Final Score: {s['final_score']:.4f}")
        print(f"    Raw: {s['raw_metrics']}")

if __name__ == "__main__":
    main()
