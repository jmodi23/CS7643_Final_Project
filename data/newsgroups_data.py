from data.text_dataset import TextDataset
from sklearn.datasets import fetch_20newsgroups
import numpy as np

class NewsGroupsDataset(TextDataset):
    """20 Newsgroups dataset implementation with optional subset control."""

    def __init__(self, categories=None, subset=None, test_size=0.2, val_size=0.1, 
                 random_state=42, remove=('headers', 'footers', 'quotes'), holdout_classes=None):
        """
        Initializes the dataset loader.

        Args:
            categories (list or None): List of categories to fetch. If None, fetches all categories.
            subset (str or None): 'train' to use only training data, 'test' to use only test data, 
                                  or None to use the full dataset.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of data to use for validation.
            random_state (int): Random seed for reproducibility.
            remove (tuple): Parts of the text to remove (headers, footers, quotes).
            holdout_classes (list or None): Classes to hold out for zero-shot evaluation.
        """
        self.categories = categories
        self.subset = subset  # Controls whether to use only 'train', 'test', or full dataset
        self.remove = remove
        super().__init__(test_size, val_size, random_state, holdout_classes)
    
    def load_data(self):
        """
        Loads the 20 Newsgroups dataset based on the specified subset.

        Returns:
            texts (list): List of documents.
            labels (np.array): Corresponding numerical labels.
            target_names (list): List of category names.
        """
        if self.subset in ['train', 'test']:  
            # Load only the specified subset
            data = fetch_20newsgroups(subset=self.subset, categories=self.categories, remove=self.remove)
            return data.data, np.array(data.target), data.target_names
        else:  
            # Load full dataset (train + test)
            train_data = fetch_20newsgroups(subset='train', categories=self.categories, remove=self.remove)
            test_data = fetch_20newsgroups(subset='test', categories=self.categories, remove=self.remove)

            all_texts = train_data.data + test_data.data
            all_labels = np.concatenate([train_data.target, test_data.target])
            target_names = train_data.target_names  # Same for train & test
            
            return all_texts, all_labels, target_names