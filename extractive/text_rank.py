
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

from indicnlp.tokenize import sentence_tokenize

def extract_key_sentences(text, top_n=5, language='en'):
    """
    Extracts key sentences from the text using TextRank algorithm.
    params:
        text: Input text to summarize.
        top_n: Number of sentences to extract.
        language: Language code (e.g., 'en', 'gu', 'hi').
    returns:
        list of strings: Top-ranked sentences.
    """
    
    # 1. Start with sentence tokenization
    if language in ['gu', 'hi', 'mr', 'bn', 'ta', 'te', 'kn', 'ml', 'pa', 'or', 'as']:
        # Use Indic NLP Library for Indian languages
        try:
            sentences = sentence_tokenize.sentence_split(text, lang=language)
        except Exception as e:
            print(f"Error using Indic NLP: {e}. Falling back to NLTK.")
            sentences = sent_tokenize(text)
    else:
        # Default to NLTK for English and others
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            sentences = sent_tokenize(text)

    if len(sentences) == 0:
        return []

    # 2. Vectorize sentences using TF-IDF
    # TF-IDF works for other languages too, as long as whitespace tokenization roughly works or we use char n-grams.
    # For Gujarati, whitespace + punctuation splitting is okay for TF-IDF.
    # We might want to use a custom tokenizer for TfidfVectorizer for Indic scripts, 
    # but standard usually works "okay" for extractive baselines.
    
    vectorizer = TfidfVectorizer(stop_words=None) # Stopwords might not be available for all langs in sklearn
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 3. Compute Similarity Matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 4. Build Graph and Compute PageRank
    # 4. Compute PageRank natively using Numpy
    # Create the column-normalized transition matrix
    n = len(sentences)
    d = 0.85 # damping factor
    
    # Avoid divide by zero
    row_sums = similarity_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1 
    transition_matrix = similarity_matrix / row_sums[:, np.newaxis]
    
    # Power iteration
    scores = np.ones(n) / n
    for _ in range(100): # max iterations
        prev_scores = np.copy(scores)
        scores = (1 - d) / n + d * transition_matrix.T.dot(scores)
        # Check convergence
        if np.sum(np.abs(scores - prev_scores)) < 1e-6:
            break

    # 5. Sort and Select Top Sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Extract top N sentences
    top_sentences = [s for score, s in ranked_sentences[:top_n]]
    
    # Return in original order
    final_indices = sorted([sentences.index(s) for s in top_sentences])
    final_summary = [sentences[i] for i in final_indices]
    
    return final_summary

if __name__ == "__main__":
    # Test block
    sample_text = """
    Automatic summarization is the process of reducing a text document with a
    computer program in order to create a summary that retains the most important points
    of the original document. As the problem of information overload has grown, and as
    the quantity of data has increased, so has interest in automatic summarization.
    Technologies that can make a coherent summary take into account variables such as
    length, writing style and syntax. An example of the use of summarization technology
    is search engines such as Google. Document summarization is another.
    """
    print("Original Text:")
    print(sample_text)
    print("\nSummary:")
    summary = extract_key_sentences(sample_text, top_n=2)
    for s in summary:
        print("-", s)
