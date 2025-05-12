# CS7643 Final Project — Comparing DistilBERT and SBERT for Scalable Semantic Search Classification

## Overview

This repository contains code for our CS7643 Deep Learning final project, where we compare the performance of **DistilBERT** and **Sentence-BERT (SBERT)** models on scalable text classification tasks through fine-tuning and semantic search. We evaluate both approaches on three different datasets: **Amazon Reviews**, **20 Newsgroups**, and **TREC**.

We investigate both **fine-tuning classification heads** and **semantic similarity retrieval** using FAISS KNN indexing.

---

## Project Structure
```bash
CS7643_Final_Project/
├── 20newgroups/                 # Fine-tuning scripts and experiments for 20 Newsgroups
├── EDA/                         # Exploratory Data Analysis notebooks
│   └── Newsgroups/
├── checkpoints/model/eval/       # Model evaluation outputs
├── data/                         # Dataset loaders (Amazon Reviews, Newsgroups, TREC)
├── eda_outputs/                  # Processed Amazon Reviews EDA outputs
├── modules/                      # Model training modules
├── 20newgroups.ipynb             # Fine-tuning notebook for Newsgroups
├── Trec_model_tuning.ipynb        # Fine-tuning notebook for TREC
├── amazon_dataset_eda.ipynb       # Amazon Reviews EDA
├── finetune.py                    # Script to fine-tune models
├── semantic_search.py             # Semantic search with SBERT/DistilBERT
├── visualize_embeddings.py        # t-SNE visualization of embeddings
├── evaluatedmodels.py             # Scripts to load and evaluate fine-tuned models
├── requirements.txt               # Python dependencies
└── README.md                      # Project README
```

---

## Datasets

- **Amazon Reviews Subset**: 20,000 sampled product reviews
- **20 Newsgroups**: News topic categorization across four selected categories
- **TREC**: Open-domain question classification dataset

Each dataset was adapted for experiments involving both full supervision and few-shot learning scenarios (by holding out certain classes).

---

## Main Components

- **Fine-tuning** DistilBERT and SBERT models using `finetune.py`
- **Semantic retrieval** with FAISS-based KNN indexing using `semantic_search.py`
- **Embedding visualization** using t-SNE in `visualize_embeddings.py`
- **Model evaluation** scripts across different datasets

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/jmodi23/CS7643_Final_Project.git
cd CS7643_Final_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Fine-Tune a Model
Example (DistilBERT fine-tuning):
```bash
python finetune.py --model distilbert-base-uncased --dataset amazon
```
Example (SBERT fine-tuning):
```
python finetune.py --model all-MiniLM-L6-v2 --dataset newsgroups
```

### 4. Run Semantic Search
```
python semantic_search.py --dataset trec --model all-MiniLM-L6-v2
```


### 5. Visualize Embeddings
```
python visualize_embeddings.py --dataset amazon --model distilbert-base-uncased
```

All outputs will be saved in the images/ directory.


## Team Members
* Jainil Modi — Amazon Reviews dataset, semantic search experiments, fine-tuning analysis
* Aaron Rodrigues — Project planning, reusable training modules, evaluation pipelines
* Joshua Belot — 20Newsgroups fine-tuning experiments + semantic search experiments
* Arun K Tipingiri — TREC fine-tuning experiments + semantic search experiments
