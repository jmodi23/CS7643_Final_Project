from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    # Text embedding model using SentenceTransformers
    # Encodes text documents into dense vector representations
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name) # Update model_name if you finetuned a model
    
    def encode_texts(self, texts):
        # Encode list of texts into normalized embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    def encode_query(self, text):
        # Encode single query text into normalized embedding
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding / np.linalg.norm(embedding)