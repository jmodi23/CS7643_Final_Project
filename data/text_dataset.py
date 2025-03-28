import numpy as np
from sklearn.model_selection import train_test_split

class TextDataset:
    # Abstract base class for text classification datasets
    def __init__(self, test_size=0.2, val_size=0.1, random_state=42, holdout_classes=None):
        # Initialize the dataset and split it into training, validation, and test sets
        self.texts, self.labels, self.label_names = self.load_data()
        self.holdout_classes = holdout_classes if holdout_classes else []
        self.holdout_class_indices = [self.label_names.index(cls) for cls in self.holdout_classes if cls in self.label_names]
        
        # Separate holdout data
        if self.holdout_class_indices:
            holdout_mask = np.isin(self.labels, self.holdout_class_indices)
            self.holdout_texts = np.array(self.texts)[holdout_mask]
            self.holdout_labels = self.labels[holdout_mask]
            # Remove holdout classes from training data
            train_mask = ~holdout_mask
            train_texts = np.array(self.texts)[train_mask]
            train_labels = self.labels[train_mask]
        else:
            train_texts = self.texts
            train_labels = self.labels
            self.holdout_texts = []
            self.holdout_labels = np.array([])
    
        # Standard train/val/test split on non-holdout data
        self.train_texts, self.test_texts, self.train_labels, self.test_labels = train_test_split(
            train_texts, train_labels, test_size=test_size, random_state=random_state)
        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
            self.train_texts, self.train_labels, test_size=val_size, random_state=random_state)
        
    def load_data(self):
        """
        Abstract method to be implemented by subclasses.
        Should return (texts, labels, label_names)
        """
        raise NotImplementedError("Subclasses to implement load_data")
    
    def get_splits(self):
        # Get the dataset splits for training, validation, and testing
        return ((self.train_texts, self.train_labels), 
                (self.val_texts, self.val_labels), 
                (self.test_texts, self.test_labels))
    
    def get_holdout_data(self):
        # Get the holdout data
        return (self.holdout_texts, self.holdout_labels)