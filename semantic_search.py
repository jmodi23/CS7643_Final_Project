from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class TextDataset:
    # Abstract base class for text classification datasets
    def __init__(self, test_size=0.2, val_size=0.1, random_state=42):
        # Initialize the dataset and split it into training, validation, and test sets
        self.texts, self.labels, self.label_names = self.load_data()
        self.train_texts, self.test_texts, self.train_labels, self.test_labels = train_test_split(
            self.texts, self.labels, test_size=test_size, random_state=random_state)
        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
            self.train_texts, self.train_labels, test_size=val_size, random_state=random_state)
    
    def load_data(self):
        """
        Abstract method to be implemented by subclasses.
        Should return (texts, labels, label_names)
        """
        raise NotImplementedError("Subclasses to implement load_data")
    
    def get_splits(self):
        # Get the dataset splits for training, validation, and testing
        return ((self.train_texts, self.train_labels), 
                (self.val_texts, self.val_labels), 
                (self.test_texts, self.test_labels))

class NewsGroupsDataset(TextDataset):
    # 20 Newsgroups dataset implementation
    def __init__(self, categories, test_size=0.2, val_size=0.1, random_state=42, subset='train', remove=('headers', 'footers', 'quotes')):
        self.categories = categories
        self.subset = subset
        self.remove = remove
        super().__init__(test_size, val_size, random_state)
    
    def load_data(self):
        data = fetch_20newsgroups(subset=self.subset, categories=self.categories, remove=self.remove)
        return data.data, np.array(data.target), data.target_names

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

class FaissIndex:
    # FAISS index for similarity search and classification
    def __init__(self, embeddings, labels):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.labels = labels
    
    def knn_search(self, query_embedding, k=5):
        # k-nearest neighbors seach for a query embedding.
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
    
    def classify(self, query_embedding, k=5):
        # Classify query embedding using KNN
        _, indices = self.knn_search(query_embedding, k)
        neighbor_labels = self.labels[indices[0]]
        return np.bincount(neighbor_labels).argmax()

def evaluate_model(embedding_model, faiss_index, test_texts, test_labels, label_names):
    # Evaluate the classification model on test data
    test_embeddings = embedding_model.encode_texts(test_texts)
    predictions = [faiss_index.classify(test_embeddings[i].reshape(1, -1)) for i in range(len(test_texts))]
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=label_names)
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

def main(dataset_class, dataset_params, model_choice='sbert'):
    # Main function to run the text classification pipeline
    dataset = dataset_class(**dataset_params)
    (train_texts, train_labels), (_, _), (test_texts, test_labels) = dataset.get_splits()
    
    model_name = 'all-MiniLM-L6-v2' if model_choice == 'sbert' else 'distilbert-base-nli-stsb-mean-tokens'
    embedding_model = EmbeddingModel(model_name)
    embeddings = embedding_model.encode_texts(train_texts)
    faiss_index = FaissIndex(embeddings, train_labels)
    
    print(f"FAISS index created w/ dimension: {embeddings.shape[1]}")
    evaluate_model(embedding_model, faiss_index, test_texts, test_labels, dataset.label_names)
    
if __name__ == "__main__":
    newsgroups_params = {
        'categories': ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast']
    }
    
    main(NewsGroupsDataset, newsgroups_params, model_choice='sbert')
    main(NewsGroupsDataset, newsgroups_params, model_choice='distilbert')