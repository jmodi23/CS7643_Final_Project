import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import get_scheduler, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------- SETTINGS ---------
MODEL_NAME_SBERT = 'all-MiniLM-L6-v2'
MODEL_NAME_DISTILBERT = 'distilbert-base-uncased'
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_PATH_SBERT = 'finetuned-sbert-model'
OUTPUT_PATH_DISTILBERT = 'finetuned-distilbert-model'

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD DATA ---------
print("Loading sampled dataset...")
df = pd.read_csv('eda_outputs/amazon_reviews_subset_20k.csv')

# Rename if needed
if 'main_category' in df.columns:
    df.rename(columns={'main_category': 'label'}, inplace=True)
if 'reviewText' in df.columns:
    df.rename(columns={'reviewText': 'text'}, inplace=True)

# -------- PREPARE DATA ---------
print("\nPreparing data for model fine-tuning...")
df = df[df['text'].notnull() & df['label'].notnull()]
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
)

# ====================== SBERT =========================
print("\nFine-tuning SBERT...")
train_examples = [
    InputExample(texts=[text, ""], label=float(label))
    for text, label in zip(train_texts, train_labels)
]

sbert_model = SentenceTransformer(MODEL_NAME_SBERT, device=device)
sbert_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
sbert_loss = losses.CosineSimilarityLoss(model=sbert_model)
sbert_warmup_steps = int(len(sbert_dataloader) * NUM_EPOCHS * 0.1)

sbert_model.fit(
    train_objectives=[(sbert_dataloader, sbert_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=sbert_warmup_steps,
    output_path=OUTPUT_PATH_SBERT,
    show_progress_bar=True
)

print("\nSBERT model fine-tuned and saved to:", OUTPUT_PATH_SBERT)

# ====================== DISTILBERT =========================
print("\nFine-tuning DistilBERT...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_DISTILBERT)

def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=256)

distilbert_dataset = Dataset.from_dict({
    "text": train_texts + test_texts,
    "label": train_labels + test_labels
})
distilbert_dataset = distilbert_dataset.train_test_split(test_size=len(test_texts))
distilbert_dataset = distilbert_dataset.map(tokenize_function, batched=True)
distilbert_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME_DISTILBERT, num_labels=len(le.classes_)).to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH_DISTILBERT,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_strategy="no",
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=distilbert_dataset['train'],
    eval_dataset=distilbert_dataset['test'],
    compute_metrics=compute_metrics
)

trainer.train()
print("\nDistilBERT model fine-tuned and saved to:", OUTPUT_PATH_DISTILBERT)
