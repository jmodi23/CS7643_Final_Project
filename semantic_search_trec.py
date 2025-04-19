import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data.newsgroups_data import NewsGroupsDataset
<<<<<<< HEAD:semantic_search.py
from .modules.embedding_model import EmbeddingModel
=======

from sklearn.utils.multiclass import unique_labels  # AT added to support mismatch in Label count for classification report

from embedding_model import EmbeddingModel
>>>>>>> atipingiri_dev:semantic_search_trec.py
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
        # k-nearest neighbors search for a query embedding.
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
    # Evaluate the classification model on test data using KNN-based prediction
    test_embeddings = embedding_model.encode_texts(test_texts)
    
    preds = []
    top_k_preds = []
    
    for i in range(len(test_texts)):
        query_emb = test_embeddings[i].reshape(1, -1)
        preds.append(faiss_index.classify(query_emb, k=k))
        top_k_preds.append(faiss_index.get_top_k_predictions(query_emb, k))
    
    # Compute standard metrics
    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, average='weighted')
    recall = recall_score(test_labels, preds, average='weighted') 
    f1 = f1_score(test_labels, preds, average='weighted')
    
    # Compute top-k accuracy
    topk_correct = 0
    for i, label in enumerate(test_labels):
        if label in top_k_preds[i]:
            topk_correct += 1
    topk_accuracy = topk_correct / len(test_labels)
    
    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for i, label in enumerate(test_labels):
        try:
            rank = top_k_preds[i].index(label) + 1  # rank starts at 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0)
    mrr = np.mean(reciprocal_ranks)
<<<<<<< HEAD:semantic_search.py
    
    # Print KNN evaluation results
    print(f"\n== {name} Evaluation Results (KNN) ==")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"MRR:         {mrr:.4f}")
=======


    active_labels = unique_labels(test_labels, preds)  # AT only labels present in y_true or y_pred


    # Print results
    print(f"\n== {name} Evaluation Results ==")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MRR: {mrr:.4f}")
>>>>>>> atipingiri_dev:semantic_search_trec.py
    print(f"Top-{k} Acc: {topk_accuracy:.4f}")
    print("\nFull Classification Report:")

    #print(classification_report(test_labels, preds, target_names=label_names))
    print(classification_report(test_labels, preds, labels = active_labels,  zero_division = 0))



    
    return preds

def train_linear_classifier(train_embeddings, train_labels_mapped, num_classes, epochs=10, learning_rate=1e-3):
    """
    Trains a simple linear classifier on top of the provided train_embeddings.
    Expects train_labels_mapped to have values in the range [0, num_classes-1].
    """
    train_emb_tensor = torch.tensor(train_embeddings, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels_mapped, dtype=torch.long)
    embedding_dim = train_embeddings.shape[1]
    
    classifier = nn.Linear(embedding_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier(train_emb_tensor)
        loss = criterion(logits, train_labels_tensor)
        loss.backward()
        optimizer.step()
        # Optionally print training loss per epoch:
        # print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
    return classifier

def compute_classifier_loss_and_perplexity(classifier, embedding_model, test_texts, test_labels_mapped):
    """
    Computes the average cross-entropy loss and perplexity (exp(loss)) for the test set using the classifier.
    Expects test_labels_mapped to have values in the same mapping as training.
    """
    classifier.eval()
    test_embeddings = embedding_model.encode_texts(test_texts)  # returns a numpy array
    test_emb_tensor = torch.tensor(test_embeddings, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels_mapped, dtype=torch.long)
    
    with torch.no_grad():
        logits = classifier(test_emb_tensor)
        loss = F.cross_entropy(logits, test_labels_tensor)
    avg_loss = loss.item()
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def run_text_classification_pipeline(dataset, model_name):
    # Main function to run the text classification pipeline
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = dataset.get_splits()
    (holdout_example_texts, holdout_example_labels), (holdout_test_texts, holdout_test_labels) = dataset.get_holdout_data()
    
    # Get label names for KNN evaluation as before
    train_label_names = [dataset.label_names[i] for i in np.unique(train_labels)]
    holdout_label_names = [dataset.label_names[i] for i in np.unique(holdout_example_labels)]
<<<<<<< HEAD:semantic_search.py
    
    print(f"Trained with classes: {', '.join(train_label_names)}")
    if len(holdout_label_names) > 0:
        print(f"Holding out classes: {', '.join(holdout_label_names)}")
    
=======


    print(f"Trained with classes: {', '.join(train_label_names)}")
    if len(holdout_label_names) > 0:
        print(f"Holding out classes: {', '.join(holdout_label_names)}")

     
>>>>>>> atipingiri_dev:semantic_search_trec.py
    embedding_model = EmbeddingModel(model_name)
    train_embeddings = embedding_model.encode_texts(train_texts)
    faiss_index = FaissIndex(train_embeddings, train_labels)
    
    print(f"FAISS index created w/ dimension: {train_embeddings.shape[1]}")
    evaluate_model(embedding_model, faiss_index, test_texts, test_labels, train_label_names, name="Regular Test")
    
    # Create a mapping from original label to a contiguous range [0, num_classes-1]
    unique_train_labels = sorted(np.unique(train_labels))
    label_mapping = {orig_label: idx for idx, orig_label in enumerate(unique_train_labels)}
    num_classes = len(label_mapping)
    
    # Remap labels for classifier training
    train_labels_mapped = np.array([label_mapping[label] for label in train_labels])
    test_labels_mapped = np.array([label_mapping[label] for label in test_labels])
    
    # Train a simple linear classifier on top of the embeddings to compute loss and perplexity.
    classifier = train_linear_classifier(train_embeddings, train_labels_mapped, num_classes, epochs=10, learning_rate=1e-3)
    clf_avg_loss, clf_perplexity = compute_classifier_loss_and_perplexity(classifier, embedding_model, test_texts, test_labels_mapped)
    
    print(f"\n== Classifier Evaluation Metrics ==")
    print(f"Average Loss: {clf_avg_loss:.4f}")
    print(f"Perplexity:   {clf_perplexity:.4f}")
    
    # Holdout Tests (if available)
    if len(holdout_example_texts) > 0:
        print("Few-Shot Learning: Adding Holdout Classes to Index")
        holdout_embeddings = embedding_model.encode_texts(holdout_example_texts)
        updated_index = FaissIndex(
            np.vstack([train_embeddings, holdout_embeddings]),
            np.hstack([train_labels, holdout_example_labels])
        )
        
        evaluate_model(
            embedding_model,
            updated_index, 
            np.hstack([test_texts, holdout_test_texts]), 
            np.hstack([test_labels, holdout_test_labels]), 
            dataset.label_names, 
            name="Holdout (Few-Shot)"
        )

# Example call:
# run_text_classification_pipeline(dataset, model_name='all-MiniLM-L6-v2')
# Example call:
# run_text_classification_pipeline(dataset, model_name='all-MiniLM-L6-v2')


# import numpy as np
# from data.newsgroups_data import NewsGroupsDataset
# from .embedding_model import EmbeddingModel
# from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
# from collections import Counter
# import faiss

# np.random.seed(10)  # Set seed
    
<<<<<<< HEAD:semantic_search.py
# class FaissIndex:
#     # FAISS index for similarity search and classification
#     def __init__(self, embeddings, labels):
#         self.index = faiss.IndexFlatL2(embeddings.shape[1])
#         self.index.add(embeddings)
#         self.labels = labels
    
#     def knn_search(self, query_embedding, k=5):
#         # k-nearest neighbors seach for a query embedding.
#         distances, indices = self.index.search(query_embedding, k)
#         return distances, indices
    
#     def classify(self, query_embedding, k=5):
#         # Classify query embedding using KNN
#         _, indices = self.knn_search(query_embedding, k)
#         neighbor_labels = self.labels[indices[0]]
#         return np.bincount(neighbor_labels).argmax()

#     def get_top_k_predictions(self, query_embedding, k=5):
#         # Returns the top k predictions based on neighbors
#         _, indices = self.knn_search(query_embedding, k)
#         neighbor_labels = self.labels[indices[0]]
#         counts = Counter(neighbor_labels)
#         # Sort by count, return k most common
#         return [label for label, _ in counts.most_common(k)]
=======
if __name__ == "__main__":


    #dataset = NewsGroupsDataset(**newsgroups_params)
    from data.trec_data import TrecDataset

    # Hold out specific fine-grained labels for few-shot generalization
    trec_params = {
        'holdout_classes': ['ENTY:currency', 'ENTY:religion', 'NUM:ord', 'NUM:temp', 'ENTY:letter', 'NUM:code', 'NUM:speed',
'ENTY:instru', 'ENTY:symbol', 'NUM:weight', 'ENTY:plant', 'NUM:volsize', 'ABBR:abb', 'ENTY:body', 'ENTY:lang', 'LOC:mount',
'HUM:title', 'ENTY:word','ENTY:veh', 'NUM:perc', 'NUM:dist', 'ENTY:techmeth', 'ENTY:color', 'ENTY:substance','ENTY:product',
'HUM:desc',
 ]  # You can change this list
    }

    dataset = TrecDataset(**trec_params)

>>>>>>> atipingiri_dev:semantic_search_trec.py

# def evaluate_model(embedding_model, faiss_index, test_texts, test_labels, label_names, k=5, name="Test"):
#     # Evaluate the classification model on test data
#     test_embeddings = embedding_model.encode_texts(test_texts)
    
#     # Get predictions and top-k predictions
#     preds = []
#     top_k_preds = []
    
#     for i in range(len(test_texts)):
#         query_emb = test_embeddings[i].reshape(1, -1)
#         preds.append(faiss_index.classify(query_emb))
#         top_k_preds.append(faiss_index.get_top_k_predictions(query_emb, 5))
    
#     # Standard metrics
#     accuracy = accuracy_score(test_labels, preds)
#     precision = precision_score(test_labels, preds, average='weighted')
#     recall = recall_score(test_labels, preds, average='weighted') 
#     f1 = f1_score(test_labels, preds, average='weighted')
    
#     # Top-k accuracy
#     topk_correct = 0
#     for i, label in enumerate(test_labels):
#         if label in top_k_preds[i]:
#             topk_correct += 1
#     topk_accuracy = topk_correct / len(test_labels)
    
#     # Mean Reciprocal Rank
#     reciprocal_ranks = []
#     for i, label in enumerate(test_labels):
#         try:
#             # Rank of true label in preds
#             rank = top_k_preds[i].index(label) + 1
#             reciprocal_ranks.append(1.0/rank)
#         except ValueError:
#             # True label not in top preds
#             reciprocal_ranks.append(0)
#     mrr = np.mean(reciprocal_ranks)
    
#     # Print results
#     print(f"\n== {name} Evaluation Results ==")
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"F1-Score: {f1:.4f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"MRR: {mrr:.4f}")
#     print(f"Top-{k} Acc: {topk_accuracy:.4f}")
#     print("\nFull Classification Report:")
#     print(classification_report(test_labels, preds, target_names=label_names))
    
#     return preds

# def run_text_classification_pipeline(dataset, model_name):
#     # Main function to run the text classification pipeline
#     (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = dataset.get_splits()
#     (holdout_example_texts, holdout_example_labels), (holdout_test_texts, holdout_test_labels) = dataset.get_holdout_data()
    
#     train_label_names = [dataset.label_names[i] for i in np.unique(train_labels)]
#     holdout_label_names = [dataset.label_names[i] for i in np.unique(holdout_example_labels)]

#     print(f"Trained with classes: {', '.join(train_label_names)}")
#     if len(holdout_label_names) > 0:
#         print(f"Holding out classes: {', '.join(holdout_label_names)}")
     
#     embedding_model = EmbeddingModel(model_name)
#     train_embeddings = embedding_model.encode_texts(train_texts)
#     faiss_index = FaissIndex(train_embeddings, train_labels)
    
#     print(f"FAISS index created w/ dimension: {train_embeddings.shape[1]}")
#     evaluate_model(embedding_model, faiss_index, test_texts, test_labels, train_label_names, name="Regular Test")

#     # Holdout Tests
#     if len(holdout_example_texts) > 0:
#         print("Few-Shot Learning: Adding Holdout Classes to Index")

#         # Update index
#         holdout_embeddings = embedding_model.encode_texts(holdout_example_texts)
#         updated_index = FaissIndex(
#             np.vstack([train_embeddings, holdout_embeddings]),
#             np.hstack([train_labels, holdout_example_labels])
#         )
        
#         # Evaluate only on the holdout test set not used in updating the index
#         evaluate_model(embedding_model, updated_index, 
#                         np.hstack([test_texts, holdout_test_texts]), 
#                         np.hstack([test_labels, holdout_test_labels]), 
#                         dataset.label_names, name="Holdout (Few-Shot)")
    
# # if __name__ == "__main__":
# #     newsgroups_params = {
# #         'categories': ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast'],
# #         'holdout_classes': ['talk.politics.mideast']
# #     }

# #     dataset = NewsGroupsDataset(**newsgroups_params)

# #     # BASE MODELS
# #     main(dataset, model_name='all-MiniLM-L6-v2')
# #     main(dataset, model_name='distilbert-base-nli-stsb-mean-tokens')