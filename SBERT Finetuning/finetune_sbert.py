import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import numpy as np


class SBERT_tuner():
    
    
    def __init__(self, module = 'all-MiniLM-L6-v2', seed = 42):
        self.seed = seed
        self.module = module
        self.set_seed(seed)
        
        pass
    
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
            
    def create_training_data(texts, labels, num_samples=10000):
        """
        Function to create training data with random positive and negative pairs for sentence embeddings.
        This function assumes that positive pairs are two samples from the same label and negative pairs 
        are samples from different labels.
        
        Args:
        texts (list): List of text samples (documents).
        labels (list): List of corresponding labels for the text samples.
        num_samples (int): The number of training examples to generate. Default is 10,000.
        
        Returns:
        train_examples (list): List of InputExample instances for training.
        """
        # Create a dictionary where the key is the label and the value is a list of texts for that label.
        label_to_texts = {}
        for text, label in zip(texts, labels):
            label_to_texts.setdefault(label, []).append(text)

        train_examples = []
        unique_labels = list(label_to_texts.keys())

        # Shuffle the text samples within each label to ensure randomness
        for label in unique_labels:
            random.shuffle(label_to_texts[label]) 

        # Generate pairs for training
        for _ in range(num_samples):
            # Randomly choose a positive label and a negative label
            pos_label = random.choice(unique_labels)
            neg_label = random.choice([l for l in unique_labels if l != pos_label])

            # Shuffle again before selecting to avoid repeated sampling patterns
            random.shuffle(label_to_texts[pos_label])
            random.shuffle(label_to_texts[neg_label])

            # Randomly select two samples from the positive label for a positive pair
            pos_pair = random.sample(label_to_texts[pos_label], 2)

            # Randomly select one sample from the negative label for a negative sample
            neg_sample = random.choice(label_to_texts[neg_label])

            # Append both a positive pair and a negative pair to the training examples
            train_examples.append(InputExample(texts=pos_pair, label=1))  # Positive pair (label=1)
            train_examples.append(InputExample(texts=[pos_pair[0], neg_sample], label=0))  # Negative pair (label=0)

        return train_examples
    
    def tune_model(self):
        
        model = SentenceTransformer(self.module)
        train_loss = losses.CosineSimilarityLoss(model) 
        
        model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,  
        warmup_steps=100,
        weight_decay=0.01,
        optimizer_params={'lr': 2e-5},
        max_grad_norm=0.5
        #scheduler 5 epochs
        
        model.save('sbert_finetuned')
        
    
)
