import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset

class DistilBERTTuner:
    def __init__(self, dataset, dataset_name, model_name='distilbert-base-nli-stsb-mean-tokens', seed=42, loss_function=None):
        """
        Initializes the tuner using a dataset instance.
        The dataset should implement get_splits(), which returns:
            ((train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels))
            
        Parameters:
          - dataset: An object that provides the training, validation, and test splits.
          - dataset_name: A string to name the dataset (used to build output paths).
          - model_name: Name or path of the model to fine-tune; by default, it uses distilbert-base-nli-stsb-mean-tokens.
          - seed: Random seed for reproducibility.
          - loss_function: A custom loss function (if any) or None to default to MultipleNegativesRankingLoss.
        """
        self.seed = seed
        self.model_name = model_name
        self.dataset = dataset
        self.set_seed(seed)
        
        # Parse splits from the dataset.
        (self.train_texts, self.train_labels), (self.val_texts, self.val_labels), (self.test_texts, self.test_labels) = dataset.get_splits()
        self.dataset_name = dataset_name
        
        # Initialize the model.
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                     else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
        
        # Set loss function; default is MultipleNegativesRankingLoss if none is provided.
        self.loss_function = losses.MultipleNegativesRankingLoss(self.model) if loss_function is None else loss_function(self.model)

    def set_seed(self, seed):
        """
        Set random seed for reproducibility across numpy, random, and torch.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def create_training_pairs(self, texts, labels, num_pairs=10000):
        """
        Generate training pairs from texts using their labels.
        For each pair, two different sentences with the same label are sampled.
        
        Parameters:
          - texts: List of sentences.
          - labels: List of labels corresponding to each sentence.
          - num_pairs: Number of training pairs to generate.
          
        Returns:
          - A list of InputExample objects for training.
        """
        label_to_texts = {}
        for text, label in zip(texts, labels):
            label_to_texts.setdefault(label, []).append(text)
            
        training_pairs = []
        unique_labels = list(label_to_texts.keys())
        
        for _ in range(num_pairs):
            pos_label = random.choice(unique_labels)
            if len(label_to_texts[pos_label]) < 2:
                continue  # Skip if there aren't enough examples for this label
            pair = random.sample(label_to_texts[pos_label], 2)
            training_pairs.append(InputExample(texts=pair))
        
        random.shuffle(training_pairs)
        return training_pairs

    def fine_tune(self, batch_size=16, epochs=1, warmup_steps=100, learning_rate=2e-5,
                  weight_decay=0.01, max_grad_norm=0.5, output_parent_path='models/tuned_distilbert/'):
        """
        Fine-tune the distilbert-base-nli-stsb-mean-tokens model.
        
        Parameters:
          - batch_size (int): Training batch size.
          - epochs (int): Number of training epochs.
          - warmup_steps (int): Number of warmup steps during training.
          - learning_rate (float): Learning rate for the optimizer.
          - weight_decay (float): Weight decay coefficient.
          - max_grad_norm (float): Maximum allowed norm of the gradients.
          - output_parent_path (str): Root directory where the tuned model will be saved.
        """
        # Create training pairs.
        train_examples = self.create_training_pairs(self.train_texts, self.train_labels)
        train_dataset = SentencesDataset(train_examples, model=self.model)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=self.model.smart_batching_collate)
        
        # Define a callback to report training loss at the end of each epoch.
        def epoch_callback(avg_loss, epoch, steps):
            print(f"Epoch {epoch} ended. Average Loss = {avg_loss:.4f}")
        
        # Define output path and start training.
        output_path = os.path.join(output_parent_path, self.dataset_name)
        self.model.fit(
            train_objectives=[(train_dataloader, self.loss_function)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            optimizer_params={'lr': learning_rate},
            max_grad_norm=max_grad_norm,
            output_path=output_path,
            callback=epoch_callback
        )
        
        print(f"Model saved at {output_path}")