
from rouge_score import rouge_scorer
import bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

# Suppress some warnings from bert_score
warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(self, coherence_model_name='all-MiniLM-L6-v2'):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = None
        try:
            self.coherence_model = SentenceTransformer(coherence_model_name)
        except Exception as e:
            print(f"Error loading coherence model: {e}")
            self.coherence_model = None

    def preload_bertscore(self):
        """
        Lazily initialize the heavyweight BERTScore model once so repeated
        requests can reuse it.
        """
        if self.bert_scorer is None:
            self.bert_scorer = bert_score.BERTScorer(lang="en")
        return self.bert_scorer

    def calculate_rouge(self, reference, candidate):
        """
        Calculates ROUGE scores.
        """
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def calculate_bertscore(self, reference, candidate):
        """
        Calculates BERTScore using RoBERTa-Large for high accuracy.
        Downloads ~1.4GB model on first run, then cached permanently.
        """
        # BERTScore calculates P, R, F1 for each pair.
        # We assume single string input for reference and candidate.
        # bert_score.score expects list of candidates and list of references.
        try:
            scorer = self.preload_bertscore()
            P, R, F1 = scorer.score([candidate], [reference])
            return F1.mean().item()
        except Exception as e:
            print(f"Error in BERTScore (likely download timeout): {e}")
            return 0.0

    def calculate_coherence(self, summary_text):
        """
        Calculates coherence score based on cosine similarity of adjacent sentences.
        """
        if not self.coherence_model:
            return 0.0

        if not summary_text:
            return 0.0

        # Split into sentences (simple split or use nltk)
        # We can use nltk sent_tokenize if available, else simple split
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(summary_text)
        except:
            sentences = summary_text.split('. ')
            
        if len(sentences) < 2:
            return 1.0 # Single sentence is coherent with itself? Or 0? 
                       # Usually if it's too short, coherence is hard to judge.
                       # Let's return 1.0 or heuristic.
        
        embeddings = self.coherence_model.encode(sentences)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            
        if not similarities:
            return 0.0
            
        return np.mean(similarities)

    def evaluate(self, reference, candidate):
        """
        Runs all metrics.
        Args:
            reference (str): Reference summary (gold standard). 
                             Note: In unsupervised setting without reference, we can't use ROUGE/BERTScore against reference.
                             But this function assumes we have one (e.g. for validation).
                             If we are doing Meta-Selection during inference WITHOUT reference, we can only use Coherence 
                             and maybe similarity to source text (Coverage).
                             
                             WAIT: The plan says "Meta-Selection mechanism for optimal summary generation".
                             If this is for inference on new text, we DON'T have a reference.
                             So strictly speaking, Meta-Selection must rely on:
                             1. Coherence
                             2. Coverage (Similarity to Source Document)
                             3. Length penalty/reward?
                             
                             The implementation plan said:
                             "Final Score = 0.4 * ROUGE + 0.3 * BERTScore + 0.3 * Coherence"
                             This implies we HAVE a reference. This suggests the user wants to pick the best model 
                             DURING DEVELOPMENT or if they have references.
                             
                             But for "Hybrid Text Summarization System" that generates summaries for new text, 
                             references are unknown.
                             
                             However, maybe the "Meta-Selection" is against the *Input Text* or the *Extractive Summary* as a pseudo-reference?
                             Common technique: Use Extractive Summary as a pseudo-reference for Abstractive candidates.
                             
                             Let's support passing "source text" or "extractive summary" as 'reference' for that purpose.
        """
        rouge = self.calculate_rouge(reference, candidate)
        bert = self.calculate_bertscore(reference, candidate)
        coherence = self.calculate_coherence(candidate)
        
        return {
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
            'bert_score': bert,
            'coherence': coherence
        }


def evaluate_summary(summary, reference, evaluator=None):
    """
    Backward-compatible helper used by benchmark scripts.

    Args:
        summary: Generated summary text.
        reference: Reference summary text.
        evaluator: Optional Evaluator instance to reuse.

    Returns:
        dict: Evaluation metrics with both `bert_score` and legacy
        `bertscore` keys for older scripts.
    """
    evaluator = evaluator or Evaluator()
    metrics = evaluator.evaluate(reference, summary)
    metrics['bertscore'] = metrics['bert_score']
    return metrics

if __name__ == "__main__":
    # Test block
    evaluator = Evaluator()
    ref = "The quick brown fox jumps over the lazy dog."
    cand = "A fast brown fox jumps over a lazy dog."
    
    print("Evaluating...")
    print(f"Ref: {ref}")
    print(f"Cand: {cand}")
    
    scores = evaluator.evaluate(ref, cand)
    print("\nScores:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
