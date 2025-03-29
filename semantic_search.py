import numpy as np
from data.newsgroups_data import NewsGroupsDataset
from embedding_model import EmbeddingModel
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from collections import Counter
import faiss

np.random.seed(10)  # Set seed
    
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

    def get_top_k_predictions(self, query_embedding, k=5):
        # Returns the top k predictions based on neighbors
        _, indices = self.knn_search(query_embedding, k)
        neighbor_labels = self.labels[indices[0]]
        counts = Counter(neighbor_labels)
        # Sort by count, return k most common
        return [label for label, _ in counts.most_common(k)]

def evaluate_model(embedding_model, faiss_index, test_texts, test_labels, label_names, k=5, name="Test"):
    # Evaluate the classification model on test data
    test_embeddings = embedding_model.encode_texts(test_texts)
    
    # Get predictions and top-k predictions
    preds = []
    top_k_preds = []
    
    for i in range(len(test_texts)):
        query_emb = test_embeddings[i].reshape(1, -1)
        preds.append(faiss_index.classify(query_emb))
        top_k_preds.append(faiss_index.get_top_k_predictions(query_emb, 5))
    
    # Standard metrics
    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, average='weighted')
    recall = recall_score(test_labels, preds, average='weighted') 
    f1 = f1_score(test_labels, preds, average='weighted')
    
    # Top-k accuracy
    topk_correct = 0
    for i, label in enumerate(test_labels):
        if label in top_k_preds[i]:
            topk_correct += 1
    topk_accuracy = topk_correct / len(test_labels)
    
    # Mean Reciprocal Rank
    reciprocal_ranks = []
    for i, label in enumerate(test_labels):
        try:
            # Rank of true label in preds
            rank = top_k_preds[i].index(label) + 1
            reciprocal_ranks.append(1.0/rank)
        except ValueError:
            # True label not in top preds
            reciprocal_ranks.append(0)
    mrr = np.mean(reciprocal_ranks)
    
    # Print results
    print(f"\n== {name} Evaluation Results ==")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Top-{k} Acc: {topk_accuracy:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(test_labels, preds, target_names=label_names))
    
    return preds

def main(dataset, model_name):
    # Main function to run the text classification pipeline
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = dataset.get_splits()
    (holdout_example_texts, holdout_example_labels), (holdout_test_texts, holdout_test_labels) = dataset.get_holdout_data()
    
    train_label_names = [dataset.label_names[i] for i in np.unique(train_labels)]
    holdout_label_names = [dataset.label_names[i] for i in np.unique(holdout_example_labels)]

    print(f"Trained with classes: {', '.join(train_label_names)}")
    if len(holdout_label_names) > 0:
        print(f"Holding out classes: {', '.join(holdout_label_names)}")
     
    embedding_model = EmbeddingModel(model_name)
    train_embeddings = embedding_model.encode_texts(train_texts)
    faiss_index = FaissIndex(train_embeddings, train_labels)
    
    print(f"FAISS index created w/ dimension: {train_embeddings.shape[1]}")
    evaluate_model(embedding_model, faiss_index, test_texts, test_labels, train_label_names, name="Regular Test")

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
                        dataset.label_names, name="Holdout (Few-Shot)")
    
if __name__ == "__main__":
    newsgroups_params = {
        'categories': ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast'],
        'holdout_classes': ['talk.politics.mideast']
    }

    dataset = NewsGroupsDataset(**newsgroups_params)

    # BASE MODELS
    main(dataset, model_name='all-MiniLM-L6-v2')
    main(dataset, model_name='distilbert-base-nli-stsb-mean-tokens')