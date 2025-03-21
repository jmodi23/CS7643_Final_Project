from data.text_dataset import TextDataset
from sklearn.datasets import fetch_20newsgroups
import numpy as np

class NewsGroupsDataset(TextDataset):
    # 20 Newsgroups dataset implementation
    def __init__(self, categories, test_size=0.2, val_size=0.1, random_state=42, 
                 subset='train', remove=('headers', 'footers', 'quotes'), holdout_classes=None):
        self.categories = categories
        self.subset = subset
        self.remove = remove
        super().__init__(test_size, val_size, random_state, holdout_classes)
    
    def load_data(self):
        data = fetch_20newsgroups(subset=self.subset, categories=self.categories, remove=self.remove)
        return data.data, np.array(data.target), data.target_names