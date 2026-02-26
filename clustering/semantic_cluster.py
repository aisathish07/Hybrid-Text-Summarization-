
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class SemanticClusterer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            raise e

    def cluster_and_select(self, sentences, valid_indices=None, n_clusters=None, ratio=0.3):
        """
        Clusters the sentences and selects the one closest to each cluster center.
        
        Args:
            sentences: List of all sentences in the text.
            valid_indices: Optional list of indices to consider (e.g., from TextRank). If None, considers all.
            n_clusters: Number of sentences to select. If None, calculated based on ratio.
            ratio: Ratio of sentences to select if n_clusters is not provided.
            
        Returns:
            list: Selected representative sentences.
        """
        if not sentences:
            return []
            
        if valid_indices is None:
            # If no pre-filtering, consider all sentences
            candidates = sentences
            original_indices = list(range(len(sentences)))
        else:
            # Filter candidates based on TextRank or other pre-filtering
            candidates = [sentences[i] for i in valid_indices]
            original_indices = valid_indices
            
        if not candidates:
            return []

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(1, int(len(candidates) * ratio))
        
        # Ensure n_clusters doesn't exceed candidate count
        n_clusters = min(n_clusters, len(candidates))
        
        if n_clusters == 0:
            return []

        # Generate embeddings
        embeddings = self.model.encode(candidates)
        
        # Check if we have enough samples for clustering
        if len(candidates) <= n_clusters:
            # Just return all candidates if they are fewer than requested clusters
            return candidates

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        
        # Find the sentence closest to each cluster center
        avg_distances = []
        closest_indices_in_candidates, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        
        # Map back to original indices and sentences
        selected_sentences = []
        selected_indices = []
        
        for idx in closest_indices_in_candidates:
            original_idx = original_indices[idx]
            selected_indices.append(original_idx)
            selected_sentences.append(sentences[original_idx])
            
        # Sort by original appearance order
        final_selection = sorted(zip(selected_indices, selected_sentences), key=lambda x: x[0])
        
        return [s for _, s in final_selection]
    
    def cluster_sentences(self, sentences, n_clusters=3):
        """
        Simplified interface for clustering sentences.
        Alias for cluster_and_select with default parameters.
        
        Args:
            sentences: List of sentences
            n_clusters: Number of clusters (default: 3)
            
        Returns:
            list: Selected representative sentences
        """
        return self.cluster_and_select(sentences, n_clusters=n_clusters)

if __name__ == "__main__":
    # Test block
    text = [
        "Artificial intelligence is transforming the world.",
        "AI is changing how we live and work.",
        "Machine learning is a subset of AI.",
        "The weather is nice today.",
        "It is sunny and warm outside.",
        "Football is a popular sport.",
        "Soccer is played by millions."
    ]
    
    print("Initializing Clusterer...")
    clusterer = SemanticClusterer()
    print("Clustering sentences...")
    # We expect clusters around AI, Weather, Sports. So maybe 3 clusters.
    selected = clusterer.cluster_and_select(text, n_clusters=3)
    
    print("\nSelected Sentences:")
    for s in selected:
        print("-", s)
