import numpy as np
from gensim.models import KeyedVectors
import nltk

# Download pre-trained word vectors (Google News)
# Note: You'll need to download this separately
# wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
MODEL_PATH = 'GoogleNews-vectors-negative300.bin.gz'

class KeywordMatcher:
    def __init__(self, model_path):
        # Load pre-trained word vectors
        self.word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    def calculate_similarity(self, keywords1, keywords2):
        """
        Calculate semantic similarity between two keyword lists
        Returns a similarity score between 0 and 1
        """
        # Filter keywords that exist in the word vector model
        valid_keywords1 = [kw for kw in keywords1 if kw in self.word_vectors.key_to_index]
        valid_keywords2 = [kw for kw in keywords2 if kw in self.word_vectors.key_to_index]
        
        # If no valid keywords, return 0
        if not valid_keywords1 or not valid_keywords2:
            return 0
        
        # Calculate pairwise similarities
        similarities = []
        for kw1 in valid_keywords1:
            for kw2 in valid_keywords2:
                similarities.append(self.word_vectors.similarity(kw1, kw2))
        
        # Return average similarity
        return np.mean(similarities)

# Example usage
def match_keywords(preprocessed_keywords, database_keywords, threshold=0.5):
    # Initialize matcher
    matcher = KeywordMatcher(MODEL_PATH)
    
    matches = []
    for db_keywords in database_keywords:
        similarity = matcher.calculate_similarity(preprocessed_keywords, db_keywords)
        if similarity >= threshold:
            matches.append((db_keywords, similarity))
    
    # Sort matches by similarity
    return sorted(matches, key=lambda x: x[1], reverse=True)

# Example database
database = [
    ['blue', 'jacket', 'leather'],
    ['red', 'backpack', 'hiking'],
    ['black', 'sunglasses', 'ray-ban']
]

# Test
query_keywords = ['blue', 'coat']
results = match_keywords(query_keywords, database)
print(results)