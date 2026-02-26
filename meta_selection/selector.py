
from evaluation.metrics import Evaluator

class MetaSelector:
    def __init__(self, evaluator=None):
        if evaluator:
            self.evaluator = evaluator
        else:
            self.evaluator = Evaluator()

    def select_best_summary(self, candidates, reference_summary):
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
                
            metrics = self.evaluator.evaluate(reference_summary, candidate_text)
            
            # Weighted Score calculation
            # Weights: 0.4 ROUGE-L + 0.3 BERTScore + 0.3 Coherence
            # Note: BERTScore returns F1 which is 0-1. ROUGE-L is 0-1. Coherence is -1 to 1 (cosine sim) but usually 0-1 for text
            
            rouge_l = metrics['rougeL']
            bert_s = metrics['bert_score']
            coherence = metrics['coherence']
            
            # Normalize coherence if needed? Cosine similarity is [-1, 1].
            # For text, usually [0, 1].
            # We'll use it as is.
            
            final_score = (0.4 * rouge_l) + (0.3 * bert_s) + (0.3 * coherence)
            
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
def select_best_summary(candidates, reference_summary):
    """
    Convenience function to select best summary without instantiating MetaSelector.
    
    Args:
        candidates (dict): {model_name: summary_text}
        reference_summary (str): The extractive summary to use as reference.
        
    Returns:
        tuple: (best_model_name, best_summary_text, scores_dict)
    """
    selector = MetaSelector()
    return selector.select_best_summary(candidates, reference_summary)

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
