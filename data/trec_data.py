import os
import urllib.request
import numpy as np
import pandas as pd
from data.text_dataset import TextDataset

class TrecDataset(TextDataset):
    DOWNLOAD_URLS = [
        "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
        "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label"
    ]

    def __init__(self, dataset_dir="TREC_Dataset", test_size=0.2, val_size=0.1, random_state=42,
                 holdout_classes=None, train_file="train_5500.label", test_file="TREC_10.label"):
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file
        self._download_if_needed()
        super().__init__(test_size, val_size, random_state, holdout_classes)

    def _download_if_needed(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        for url in self.DOWNLOAD_URLS:
            file_name = os.path.join(self.dataset_dir, url.split("/")[-1])
            if not os.path.exists(file_name):
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(url, file_name)
            else:
                print(f"Found existing file: {file_name}")

    def load_data(self):
        train_df = self._load_trec_data(os.path.join(self.dataset_dir, self.train_file))
        test_df = self._load_trec_data(os.path.join(self.dataset_dir, self.test_file))
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

        # Create label mappings
        label_names = sorted(full_df['label'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(label_names)}
        labels = np.array([label_to_idx[label] for label in full_df['label']])
        texts = full_df['question'].tolist()
        return texts, labels, label_names

    def _load_trec_data(self, file_path):
        labels = []
        questions = []
        with open(file_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    labels.append(parts[0])
                    questions.append(parts[1])
        return pd.DataFrame({"label": labels, "question": questions})

