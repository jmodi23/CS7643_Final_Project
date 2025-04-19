# Amazon Product Reviews - Multi-Category Subset, EDA & Data Preparation + SBERT & DistilBERT Embedding Extraction + Classifier

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss, precision_score, recall_score, f1_score
import gzip
import random
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

# -------- SETTINGS ---------
TOTAL_RECORDS = 20000
OUTPUT_DIR = 'eda_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Categories and files to load
CATEGORIES = [
    'Electronics',
    'Books',
    'Home_and_Kitchen',
    'Clothing_Shoes_and_Jewelry',
    'Sports_and_Outdoors'
]

DATA_FOLDER = 'amazon_reviews_data'
os.makedirs(DATA_FOLDER, exist_ok=True)

# -------- STREAM JSON.GZ LINES & COLLECT DATA ---------

def stream_collect_json(category, max_records=None):
    gz_file_path = os.path.join(DATA_FOLDER, f'{category}.json.gz')

    if not os.path.exists(gz_file_path):
        raise FileNotFoundError(f"Compressed JSON file {gz_file_path} not found.")

    print(f"Streaming and collecting from: {gz_file_path}")

    records = []
    with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            records.append(record)
            if max_records and len(records) >= max_records:
                break

    df = pd.DataFrame(records)
    df['main_category'] = category
    return df

# Parallel streaming and collecting
def parallel_collect(categories, max_records_per_category=None):
    df_list = []
    with ThreadPoolExecutor(max_workers=len(categories)) as executor:
        futures = {
            executor.submit(stream_collect_json, cat, max_records_per_category): cat
            for cat in categories
        }
        for future in as_completed(futures):
            category = futures[future]
            try:
                df = future.result()
                df_list.append(df)
                print(f"Finished collecting for category: {category}, records: {df.shape[0]}")
            except Exception as exc:
                print(f"Category {category} generated an exception: {exc}")
    return df_list

# Collect large dataset first
MAX_RECORDS_PER_CATEGORY = 100000
df_list = parallel_collect(CATEGORIES, MAX_RECORDS_PER_CATEGORY)

# Combine all categories into one dataframe
df_combined = pd.concat(df_list, ignore_index=True)
print(f"Total combined dataset shape: {df_combined.shape}")

# Random sample from the combined dataset
df_sampled = df_combined.sample(n=min(TOTAL_RECORDS, len(df_combined)), random_state=42).reset_index(drop=True)
print(f"Randomly sampled dataset shape: {df_sampled.shape}")

# Save sampled dataset immediately
SAMPLED_DATA_FILE = os.path.join(OUTPUT_DIR, 'amazon_reviews_subset_20k.csv')
df_sampled.to_csv(SAMPLED_DATA_FILE, index=False)
print(f"Sampled dataset saved at: {SAMPLED_DATA_FILE}")

# Rename columns
column_rename_map = {
    'reviewText': 'text',
    'main_category': 'label',
    'overall': 'rating'
}
existing_cols = df_sampled.columns.tolist()
rename_dict = {k: v for k, v in column_rename_map.items() if k in existing_cols}
df_sampled.rename(columns=rename_dict, inplace=True)

print("\nSample records:")
print(df_sampled[['text', 'rating', 'label']].head())

# -------- EDA ---------

EDA_FOLDER = os.path.join(OUTPUT_DIR, 'eda_charts')
os.makedirs(EDA_FOLDER, exist_ok=True)

plt.figure(figsize=(10,6))
df_sampled['label'].value_counts().plot(kind='bar')
plt.title("Distribution of Categories in Sampled Dataset")
plt.xlabel("Category")
plt.ylabel("Number of Samples")
plt.savefig(f"{EDA_FOLDER}/category_distribution.png")
plt.close()

df_sampled['char_length'] = df_sampled['text'].astype(str).apply(len)
df_sampled['word_count'] = df_sampled['text'].astype(str).apply(lambda x: len(x.split()))

avg_seq_len = df_sampled['char_length'].mean()
avg_word_count = df_sampled['word_count'].mean()

plt.figure(figsize=(10,6))
sns.histplot(df_sampled['char_length'], bins=50)
plt.title("Character Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.savefig(f"{EDA_FOLDER}/char_length_distribution.png")
plt.close()

plt.figure(figsize=(10,6))
sns.histplot(df_sampled['word_count'], bins=50)
plt.title("Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig(f"{EDA_FOLDER}/word_count_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(x='rating', data=df_sampled)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.savefig(f"{EDA_FOLDER}/ratings_distribution.png")
plt.close()

text_data = " ".join(df_sampled['text'].astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Review Text")
plt.savefig(f"{EDA_FOLDER}/word_cloud.png")
plt.close()

all_words = " ".join(df_sampled['text'].astype(str).tolist()).split()
vocab_size = len(set(all_words))

class_distribution = df_sampled['label'].value_counts().to_dict()
num_classes = len(class_distribution)
max_class = max(class_distribution.values())
min_class = min(class_distribution.values())
skewness = "Highly Skewed" if max_class > 2 * min_class else "Balanced"

label_overlap = len(set(df_sampled['label'].tolist()))
topic_complexity = "High" if num_classes > 20 else "Medium" if num_classes > 10 else "Low"

print("\n===== EDA SUMMARY =====")
print(f"Average Sequence Length (chars): {avg_seq_len:.2f}")
print(f"Average Word Count: {avg_word_count:.2f}")
print(f"Number of Classes: {num_classes}")
print(f"Class Distribution: {class_distribution}")
print(f"Skewness: {skewness}")
print(f"Vocabulary Size: {vocab_size}")
print(f"Label Overlap: {label_overlap}")
print(f"Topic Complexity: {topic_complexity}")

# -------- PREPARE DATA FOR EMBEDDINGS ---------

# Filter out empty or null texts first
prepared_df = df_sampled[['text', 'label']].copy()
prepared_df['text'] = prepared_df['text'].astype(str).str.strip()
prepared_df = prepared_df[prepared_df['text'].notnull() & (prepared_df['text'] != '')].copy()

# Encode labels
le = LabelEncoder()
prepared_df['label_encoded'] = le.fit_transform(prepared_df['label'])

prepared_file = os.path.join(OUTPUT_DIR, 'amazon_reviews_prepared_20k.csv')
prepared_df.to_csv(prepared_file, index=False)
print("Prepared dataset saved at:", prepared_file)

from sklearn.model_selection import train_test_split

# -------- SBERT EMBEDDINGS + CLASSIFIER ---------

print("Encoding sentences with SBERT...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
sbert_start_time = time.time()
sbert_embeddings = sbert_model.encode(prepared_df['text'].tolist(), batch_size=128, show_progress_bar=True, convert_to_numpy=True)
print(f"Finished encoding with SBERT in {time.time() - sbert_start_time:.2f} seconds.")

X_train_sbert, X_test_sbert, y_train_sbert, y_test_sbert = train_test_split(sbert_embeddings, prepared_df['label_encoded'], test_size=0.2, random_state=42)

sbert_classifier = LogisticRegression(max_iter=1000)
sbert_classifier.fit(X_train_sbert, y_train_sbert)

y_pred_sbert = sbert_classifier.predict(X_test_sbert)
sbert_acc = accuracy_score(y_test_sbert, y_pred_sbert)
sbert_precision = precision_score(y_test_sbert, y_pred_sbert, average='macro')
sbert_recall = recall_score(y_test_sbert, y_pred_sbert, average='macro')
sbert_f1 = f1_score(y_test_sbert, y_pred_sbert, average='macro')
sbert_probs = sbert_classifier.predict_proba(X_test_sbert)
sbert_loss = log_loss(y_test_sbert, sbert_probs)
sbert_perplexity = np.exp(sbert_loss)

print("SBERT Classification Report:")
print(classification_report(y_test_sbert, y_pred_sbert, target_names=le.classes_))
print(f"SBERT Accuracy: {sbert_acc:.4f}")
print(f"SBERT Precision (Macro): {sbert_precision:.4f}")
print(f"SBERT Recall (Macro): {sbert_recall:.4f}")
print(f"SBERT F1-Score (Macro): {sbert_f1:.4f}")
print(f"SBERT Log Loss: {sbert_loss:.4f}")
print(f"SBERT Perplexity: {sbert_perplexity:.4f}")

joblib.dump(sbert_classifier, os.path.join(OUTPUT_DIR, 'sbert_logistic_classifier.joblib'))
print("SBERT classifier saved at:", os.path.join(OUTPUT_DIR, 'sbert_logistic_classifier.joblib'))

# -------- DISTILBERT EMBEDDINGS + CLASSIFIER ---------

# Filter out empty or null texts first
prepared_df = df_sampled[['text', 'label']].copy()
prepared_df['text'] = prepared_df['text'].astype(str).str.strip()
prepared_df = prepared_df[prepared_df['text'].notnull() & (prepared_df['text'] != '')].copy()

# Encode labels
le = LabelEncoder()
prepared_df['label_encoded'] = le.fit_transform(prepared_df['label'])

prepared_file = os.path.join(OUTPUT_DIR, 'amazon_reviews_prepared_20k.csv')
prepared_df.to_csv(prepared_file, index=False)
print("\nPrepared dataset saved at:", prepared_file)

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()

# Tokenize and create embeddings
def get_distilbert_embeddings(text_list, batch_size=128):
    from tqdm import tqdm
    embeddings = []
    with torch.no_grad():
        cleaned_texts = [str(text).strip() if pd.notnull(text) else "" for text in text_list]
        for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Encoding Batches"):
            batch = cleaned_texts[i:i+batch_size]
            filtered_batch = [text for text in batch if text]
            if not filtered_batch:
                continue
            inputs = tokenizer(filtered_batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

print("\nEncoding sentences with DistilBERT...")
start_time = time.time()
distilbert_embeddings = get_distilbert_embeddings(prepared_df['text'].tolist())
print(f"Finished encoding in {time.time() - start_time:.2f} seconds.")

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(distilbert_embeddings, prepared_df['label_encoded'], test_size=0.2, random_state=42)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Overall Metrics
overall_precision = precision_score(y_test, y_pred, average='macro')
overall_recall = recall_score(y_test, y_pred, average='macro')
overall_f1 = f1_score(y_test, y_pred, average='macro')

# Log Loss
probs = classifier.predict_proba(X_test)
loss = log_loss(y_test, probs)

# Perplexity
perplexity = np.exp(loss)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"\nAccuracy: {acc:.4f}")
print(f"Overall Precision (Macro): {overall_precision:.4f}")
print(f"Overall Recall (Macro): {overall_recall:.4f}")
print(f"Overall F1-Score (Macro): {overall_f1:.4f}")
print(f"Log Loss: {loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")

# Save classifier (optional)
import joblib
joblib.dump(classifier, os.path.join(OUTPUT_DIR, 'distilbert_logistic_classifier.joblib'))
print("Classifier saved at:", os.path.join(OUTPUT_DIR, 'distilbert_logistic_classifier.joblib'))
