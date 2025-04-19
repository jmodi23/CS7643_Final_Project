import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data.text_dataset import TextDataset

class AmazonReviewsDataset(TextDataset):
    def __init__(self, csv_path='eda_outputs/amazon_reviews_subset_20k.csv',
                 test_size=0.2, val_size=0.1, random_state=42, holdout_classes=None):
        self.csv_path = csv_path
        self.holdout_classes = holdout_classes or []
        super().__init__(test_size, val_size, random_state, holdout_classes)

    def load_data(self):
        data = pd.read_csv(self.csv_path)

        # Patch: Rename columns if needed
        if 'reviewText' in data.columns:
            data.rename(columns={'reviewText': 'text'}, inplace=True)
        if 'main_category' in data.columns:
            data.rename(columns={'main_category': 'label'}, inplace=True)
        if 'overall' in data.columns and 'rating' not in data.columns:
            data.rename(columns={'overall': 'rating'}, inplace=True)

        # Basic validation
        if 'text' not in data.columns or 'label' not in data.columns:
            raise ValueError("Required columns 'text' and 'label' not found after renaming.")

        # Drop rows with missing data
        data = data[data['text'].notnull() & data['label'].notnull()]

        self.label_names = sorted(data['label'].unique())
        label_to_id = {label: i for i, label in enumerate(self.label_names)}
        labels = data['label'].map(label_to_id).values

        return data['text'].tolist(), labels.tolist(), self.label_names
