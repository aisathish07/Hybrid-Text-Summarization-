
from evaluation.metrics import Evaluator

class MetaSelector:
    def __init__(self, evaluator=None):
        if evaluator:
            self.evaluator = evaluator
        else:
            self.evaluator = Evaluator()

    def select_best_summary(self, candidates, reference_summary, language='en'):
        """
        Selects the best summary from candidates based on weighted scores against reference_summary.
        
        Args:
            candidates (dict): {model_name: summary_text}
            reference_summary (str): The extractive summary to use as reference.
            
        Returns:
            tuple: (best_model_name, best_summary_text, scores_dict)
        """
        best_score = -1.0
        best_model = None
        best_summary = None
        all_scores = {}
        
        for model_name, candidate_text in candidates.items():
            if not candidate_text:
                continue
                
            metrics = self.evaluator.evaluate(reference_summary, candidate_text, language=language)

            rouge1 = metrics['rouge1']
            rouge_l = metrics['rougeL']
            bert_s = metrics['bert_score']
            semantic_coverage = metrics['semantic_coverage']
            coherence = metrics['coherence']
            length_adequacy = metrics['length_adequacy']

            if language == 'en':
                final_score = (
                    (0.15 * rouge1) +
                    (0.20 * rouge_l) +
                    (0.25 * semantic_coverage) +
                    (0.20 * bert_s) +
                    (0.10 * coherence) +
                    (0.10 * length_adequacy)
                )
            else:
                final_score = (
                    (0.20 * rouge1) +
                    (0.20 * rouge_l) +
                    (0.35 * semantic_coverage) +
                    (0.15 * coherence) +
                    (0.10 * length_adequacy)
                )
            
            all_scores[model_name] = {
                'raw_metrics': metrics,
                'final_score': final_score
            }
            
            if final_score > best_score:
                best_score = final_score
                best_model = model_name
                best_summary = candidate_text
                
        return best_model, best_summary, all_scores

# Module-level convenience function
def select_best_summary(candidates, reference_summary, language='en'):
    """
    Convenience function to select best summary without instantiating MetaSelector.
    
    Args:
        candidates (dict): {model_name: summary_text}
        reference_summary (str): The extractive summary to use as reference.
        
    Returns:
        tuple: (best_model_name, best_summary_text, scores_dict)
    """
    selector = MetaSelector()
    return selector.select_best_summary(candidates, reference_summary, language=language)

if __name__ == "__main__":
    # Test block
    selector = MetaSelector()
    
    ref = "The quick brown fox jumps over the lazy dog."
    cands = {
        "model_a": "A fast brown fox jumps over a lazy dog.",
        "model_b": "The dog is lazy and the fox is brown.",
        "model_c": "Apples are red."
    }
    
    best_model, best_sum, scores = selector.select_best_summary(cands, ref)
    
    print(f"Reference: {ref}")
    print(f"Best Model: {best_model} (Score: {scores[best_model]['final_score']:.4f})")
    print(f"Best Summary: {best_sum}")
    print("\nDetailed Scores:")
    for m, s in scores.items():
        print(f"{m}: {s}")
