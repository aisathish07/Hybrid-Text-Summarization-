
import re
import warnings

import bert_score
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress some warnings from bert_score
warnings.filterwarnings("ignore")

class Evaluator:
    def __init__(
        self,
        coherence_model_name='all-MiniLM-L6-v2',
        coverage_model_name='paraphrase-multilingual-MiniLM-L12-v2',
        coverage_model=None,
    ):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = None
        self.coverage_model = coverage_model

        if self.coverage_model is None:
            try:
                self.coverage_model = SentenceTransformer(coverage_model_name)
            except Exception as e:
                print(f"Error loading coverage model: {e}")
                self.coverage_model = None

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
            'rouge1': float(scores['rouge1'].fmeasure),
            'rouge2': float(scores['rouge2'].fmeasure),
            'rougeL': float(scores['rougeL'].fmeasure)
        }

    def calculate_bertscore(self, reference, candidate, language='en'):
        """
        Calculates BERTScore using RoBERTa-Large for high accuracy.
        Downloads ~1.4GB model on first run, then cached permanently.
        """
        if language != 'en':
            return 0.0

        try:
            scorer = self.preload_bertscore()
            P, R, F1 = scorer.score([candidate], [reference])
            return F1.mean().item()
        except Exception as e:
            print(f"Error in BERTScore (likely download timeout): {e}")
            return 0.0

    def _select_sentence_encoder(self, language='en'):
        if language == 'en' and self.coherence_model:
            return self.coherence_model
        return self.coverage_model or self.coherence_model

    def _split_sentences(self, text, language='en'):
        if not text:
            return []

        if language == 'en':
            try:
                from nltk.tokenize import sent_tokenize
                return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
            except Exception:
                pass

        return [part.strip() for part in re.split(r'[.!?।]+', text) if part.strip()]

    def calculate_semantic_coverage(self, reference, candidate):
        """
        Measures how well the candidate semantically matches the reference.
        """
        if not self.coverage_model or not reference or not candidate:
            return 0.0

        try:
            embeddings = self.coverage_model.encode([reference, candidate])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, min(1.0, float((similarity + 1.0) / 2.0)))
        except Exception as e:
            print(f"Error in semantic coverage: {e}")
            return 0.0

    def calculate_length_adequacy(self, reference, candidate):
        """
        Penalizes summaries that are far shorter than the clustered reference.
        """
        reference_tokens = len(reference.split())
        candidate_tokens = len(candidate.split())

        if reference_tokens == 0 or candidate_tokens == 0:
            return 0.0

        target_min = max(8, int(reference_tokens * 0.45))
        target_max = max(target_min + 1, int(reference_tokens * 1.15))

        if candidate_tokens < target_min:
            return max(0.0, min(1.0, candidate_tokens / target_min))

        if candidate_tokens > target_max:
            return max(0.0, min(1.0, target_max / candidate_tokens))

        return 1.0

    def calculate_coherence(self, summary_text, language='en'):
        """
        Calculates coherence score based on cosine similarity of adjacent sentences.
        """
        encoder = self._select_sentence_encoder(language)
        if not encoder:
            return 0.0

        if not summary_text:
            return 0.0

        sentences = self._split_sentences(summary_text, language=language)
            
        if len(sentences) < 2:
            return 1.0
        
        embeddings = encoder.encode(sentences)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            
        if not similarities:
            return 0.0
            
        return float(np.mean(similarities))

    def evaluate(self, reference, candidate, language='en'):
        """
        Runs all metrics.
        """
        rouge = self.calculate_rouge(reference, candidate)
        bert = self.calculate_bertscore(reference, candidate, language=language)
        semantic_coverage = self.calculate_semantic_coverage(reference, candidate)
        coherence = self.calculate_coherence(candidate, language=language)
        length_adequacy = self.calculate_length_adequacy(reference, candidate)
        
        return {
            'rouge1': float(rouge['rouge1']),
            'rouge2': float(rouge['rouge2']),
            'rougeL': float(rouge['rougeL']),
            'bert_score': float(bert),
            'semantic_coverage': float(semantic_coverage),
            'coherence': float(coherence),
            'length_adequacy': float(length_adequacy),
        }


def evaluate_summary(summary, reference, evaluator=None, language='en'):
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
    metrics = evaluator.evaluate(reference, summary, language=language)
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
