import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from tqdm import tqdm

# ----- Load Dataset -----
df = pd.read_csv("eda_outputs/amazon_reviews_subset_20k.csv")

if 'main_category' in df.columns:
    df.rename(columns={'main_category': 'label'}, inplace=True)
if 'reviewText' in df.columns:
    df.rename(columns={'reviewText': 'text'}, inplace=True)

df = df[df['text'].notnull() & df['label'].notnull()]

# Encode labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])

# Train/Test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
)

# ----- Load Classifiers -----
sbert_clf = joblib.load("finetuned-sbert-model/sbert_logistic_classifier.joblib")
distilbert_clf = joblib.load("finetuned-distilbert-model/distilbert_logistic_classifier.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- SBERT Embedding -----
print("\nEncoding test set with SBERT...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
sbert_embeddings = sbert_model.encode(test_texts, convert_to_numpy=True, show_progress_bar=True)

# ----- DISTILBERT Embedding -----
print("\nEncoding test set with DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

def get_distilbert_embeddings(texts, batch_size=64):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    return np.vstack(all_embeddings)

distilbert_embeddings = get_distilbert_embeddings(test_texts)

# ===== Metrics Helper =====
def evaluate_model(preds, probs, name):
    acc = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, average='macro')
    recall = recall_score(test_labels, preds, average='macro')
    f1 = f1_score(test_labels, preds, average='macro')
    loss = log_loss(test_labels, probs)
    perplexity = np.exp(loss)

    print(f"\n=== {name} Fine-Tuned Classifier ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-Score     : {f1:.4f}")
    print(f"Log Loss     : {loss:.4f}")
    print(f"Perplexity   : {perplexity:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=le.classes_))

# ----- Evaluate SBERT -----
sbert_preds = sbert_clf.predict(sbert_embeddings)
sbert_probs = sbert_clf.predict_proba(sbert_embeddings)
evaluate_model(sbert_preds, sbert_probs, "SBERT")

# ----- Evaluate DistilBERT -----
distil_preds = distilbert_clf.predict(distilbert_embeddings)
distil_probs = distilbert_clf.predict_proba(distilbert_embeddings)
evaluate_model(distil_preds, distil_probs, "DistilBERT")
