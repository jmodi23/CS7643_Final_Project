import numpy as np
import pandas as pd
from data.amazonreviews_data import AmazonReviewsDataset
from data.newsgroups_data import NewsGroupsDataset
from data.trec_data import TrecDataset

from modules.embedding_model import EmbeddingModel
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import faiss

np.random.seed(10)

class FaissIndex:
    def __init__(self, embeddings, labels):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.labels = labels

    def knn_search(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def classify(self, query_embedding, k=5):
        _, indices = self.knn_search(query_embedding, k)
        neighbor_labels = self.labels[indices[0]]
        return np.bincount(neighbor_labels).argmax()

    def get_top_k_predictions(self, query_embedding, k=5):
        _, indices = self.knn_search(query_embedding, k)
        neighbor_labels = self.labels[indices[0]]
        counts = Counter(neighbor_labels)
        return [label for label, _ in counts.most_common(k)]

def evaluate_model(embedding_model, faiss_index, test_texts, test_labels, label_names, k=5, name="Test"):
    test_embeddings = embedding_model.encode_texts(test_texts)
    preds = []
    top_k_preds = []

    for i in range(len(test_texts)):
        query_emb = test_embeddings[i].reshape(1, -1)
        preds.append(faiss_index.classify(query_emb))
        top_k_preds.append(faiss_index.get_top_k_predictions(query_emb, k))

    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, average='weighted')
    recall = recall_score(test_labels, preds, average='weighted') 
    f1 = f1_score(test_labels, preds, average='weighted')

    topk_correct = sum([label in top_k_preds[i] for i, label in enumerate(test_labels)])
    topk_accuracy = topk_correct / len(test_labels)

    reciprocal_ranks = []
    for i, label in enumerate(test_labels):
        try:
            rank = top_k_preds[i].index(label) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0)
    mrr = np.mean(reciprocal_ranks)

    print(f"\n== {name} Evaluation Results ==")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"Top-{k} Acc: {topk_accuracy:.4f}")

    print("\nFull Classification Report:")
    labels_in_test = sorted(list(unique_labels(test_labels, preds)))
    label_names_subset = [label_names[i] for i in labels_in_test]
    print(classification_report(test_labels, preds, labels=labels_in_test, target_names=label_names_subset))

    return preds

def main(dataset, model_name):
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = dataset.get_splits()
    (holdout_example_texts, holdout_example_labels), (holdout_test_texts, holdout_test_labels) = dataset.get_holdout_data()

    train_label_names = [dataset.label_names[i] for i in np.unique(train_labels)]
    holdout_label_names = [dataset.label_names[i] for i in np.unique(holdout_example_labels)]
    
    print(f"Trained with classes: {', '.join(train_label_names)}")
    if len(holdout_label_names) > 0:
        print(f"Holding out classes: {', '.join(holdout_label_names)}")

    # Load model
    embedding_model = EmbeddingModel(model_name)
    train_embeddings = embedding_model.encode_texts(train_texts)
    faiss_index = FaissIndex(train_embeddings, train_labels)

    print(f"FAISS index created w/ dimension: {train_embeddings.shape[1]}")
    evaluate_model(embedding_model, faiss_index, test_texts, test_labels, dataset.label_names, name="Regular Test")

    if len(holdout_example_texts) > 0:
        print("Few-Shot Learning: Adding Holdout Classes to Index")
        holdout_embeddings = embedding_model.encode_texts(holdout_example_texts)
        updated_index = FaissIndex(
            np.vstack([train_embeddings, holdout_embeddings]),
            np.hstack([train_labels, holdout_example_labels])
        )
        evaluate_model(
            embedding_model, updated_index,
            np.hstack([test_texts, holdout_test_texts]),
            np.hstack([test_labels, holdout_test_labels]),
            dataset.label_names,
            name="Holdout (Few-Shot)"
        )

if __name__ == "__main__":

    datasets = ['amazon', 'newsgroups', 'trec']
    dataset_chosen = datasets[0]

    print(f"Evaluating for dataset {dataset_chosen}")
    if dataset_chosen == datasets[0]:
        params = {
            'csv_path': 'eda_outputs/amazon_reviews_subset_20k.csv',
            'holdout_classes': ['Books']
        }
        dataset = AmazonReviewsDataset(**params)

        ft_sbert_loc, ft_dtbert_loc = 'finetuned-sbert-model', 'finetuned-distilbert-model'

    elif dataset_chosen == datasets[1]:
        newsgroups_params = {
            'categories': ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast'],
            'holdout_classes': ['talk.politics.mideast']
        }
        dataset = NewsGroupsDataset(**newsgroups_params)

        ft_sbert_loc, ft_dtbert_loc = 'finetuned-sbert-model', 'finetuned-distilbert-model' # REPLACE

    elif dataset_chosen == datasets[2]:
        trec_params = {
            'holdout_classes': ['ENTY:currency', 'ENTY:religion', 'NUM:ord', 'NUM:temp', 'ENTY:letter', 'NUM:code', 'NUM:speed',
            'ENTY:instru', 'ENTY:symbol', 'NUM:weight', 'ENTY:plant', 'NUM:volsize', 'ABBR:abb', 'ENTY:body', 'ENTY:lang', 'LOC:mount',
            'HUM:title', 'ENTY:word','ENTY:veh', 'NUM:perc', 'NUM:dist', 'ENTY:techmeth', 'ENTY:color', 'ENTY:substance','ENTY:product',
            'HUM:desc',
            ]
        }
        dataset = TrecDataset(**trec_params)

        ft_sbert_loc, ft_dtbert_loc = 'finetuned-sbert-model', 'finetuned-distilbert-model' # REPLACE

    else:
        raise "Dataset not chosen!"

    print("## ------------------ Untuned Models -------------")
    main(dataset, model_name='all-MiniLM-L6-v2')
    main(dataset, model_name='distilbert-base-uncased') # Shockingly this works even though DistilBERT is not from Sentence Transformers. Note that it is doing mean pooling and not directly using the CLS token.
    # main(dataset, model_name='distilbert-base-nli-stsb-mean-tokens')

    print("\n## ------------------ Fine-Tuned Models -------------")
    # Replace model_name with the path to the finetuned models
    main(dataset, model_name=ft_sbert_loc)
    main(dataset, model_name=ft_dtbert_loc)
