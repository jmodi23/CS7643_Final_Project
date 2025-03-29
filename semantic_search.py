from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from data.newsgroups_data import NewsGroupsDataset
np.random.seed(10)  # Set seed

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

def evaluate_model(embedding_model, faiss_index, test_texts, test_labels, label_names, name="Test"):
    # Evaluate the classification model on test data
    test_embeddings = embedding_model.encode_texts(test_texts)
    predictions = [faiss_index.classify(test_embeddings[i].reshape(1, -1)) for i in range(len(test_texts))]
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=label_names)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(report)

    return predictions

def main(dataset, model_choice='sbert'):
    # Main function to run the text classification pipeline
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = dataset.get_splits()
    (holdout_example_texts, holdout_example_labels), (holdout_test_texts, holdout_test_labels) = dataset.get_holdout_data()
    
    train_label_names = [dataset.label_names[i] for i in np.unique(train_labels)]
    holdout_label_names = [dataset.label_names[i] for i in np.unique(holdout_example_labels)]

    print(f"Trained with classes: {', '.join(train_label_names)}")
    if len(holdout_label_names) > 0:
        print(f"Holding out classes: {', '.join(holdout_label_names)}")
    
    # Choose SBERT or DistilBERT
    model_name = ('all-MiniLM-L6-v2' if model_choice == 'sbert' 
                  else 'distilbert-base-nli-stsb-mean-tokens')
    embedding_model = EmbeddingModel(model_name)
    train_embeddings = embedding_model.encode_texts(train_texts)
    faiss_index = FaissIndex(train_embeddings, train_labels)
    
    print(f"FAISS index created w/ dimension: {train_embeddings.shape[1]}")
    evaluate_model(embedding_model, faiss_index, test_texts, test_labels, train_label_names, "Regular Test")

    # Holdout Tests
    if len(holdout_example_texts) > 0:
        print("Few-Shot Learning: Adding Holdout Classes to Index")

        # Update index
        holdout_embeddings = embedding_model.encode_texts(holdout_example_texts)
        updated_index = FaissIndex(
            np.vstack([train_embeddings, holdout_embeddings]),
            np.hstack([train_labels, holdout_example_labels])
        )
        
        # Evaluate only on the holdout test set not used in updating the index
        evaluate_model(embedding_model, updated_index, 
                        np.hstack([test_texts, holdout_test_texts]), 
                        np.hstack([test_labels, holdout_test_labels]), 
                        dataset.label_names, "Holdout (Few-Shot)")
    
if __name__ == "__main__":
    newsgroups_params = {
        'categories': ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast'],
        'holdout_classes': ['talk.politics.mideast']
    }

    main(NewsGroupsDataset(**newsgroups_params), model_choice='sbert')
    main(NewsGroupsDataset(**newsgroups_params), model_choice='distilbert')