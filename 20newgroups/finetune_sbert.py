import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
import torch.nn.functional as F

# ---------------------------
# Student-Teacher Loss Class
# ---------------------------
class StudentTeacherLoss(nn.Module):
    """
    A composite loss that adds a KL divergence term between the student and teacher model outputs
    (measured on their pairwise similarity distributions) to the base loss.
    
    Args:
        student_model: The model being fine-tuned.
        teacher_model: A frozen model (copy of the original) used as teacher.
        base_loss: The original loss (e.g. MultipleNegativesRankingLoss) used for the task.
        temperature: Temperature for softening distributions.
        alpha: Weight factor for the base loss (the remaining weight, 1-alpha, is for the KL divergence term).
    """
    def __init__(self, student_model, teacher_model, base_loss, temperature=2.0, alpha=0.7):
        super(StudentTeacherLoss, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.base_loss = base_loss
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, sentence_features, labels=None):
        # 1. Compute the base loss using the wrapped loss function.
        base_loss_value = self.base_loss(sentence_features, labels)
        
        # 2. Compute student embeddings by forwarding each feature dict through the student model.
        student_embeddings_list = []
        for features in sentence_features:
            out = self.student_model(features)
            # The SentenceTransformer model returns a dict with the key "sentence_embedding"
            embedding = out['sentence_embedding'] if isinstance(out, dict) and 'sentence_embedding' in out else out
            student_embeddings_list.append(embedding)
        student_embeddings = torch.vstack(student_embeddings_list)
        
        # 3. Compute teacher embeddings (no gradient) similarly.
        with torch.no_grad():
            teacher_embeddings_list = []
            for features in sentence_features:
                out = self.teacher_model(features)
                embedding = out['sentence_embedding'] if isinstance(out, dict) and 'sentence_embedding' in out else out
                teacher_embeddings_list.append(embedding)
            teacher_embeddings = torch.vstack(teacher_embeddings_list)
        
        # 4. Normalize embeddings and compute pairwise cosine similarity matrices.
        student_norm = F.normalize(student_embeddings, p=2, dim=1)
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1)
        student_sim = torch.matmul(student_norm, student_norm.transpose(0, 1)) / self.temperature
        teacher_sim = torch.matmul(teacher_norm, teacher_norm.transpose(0, 1)) / self.temperature
        
        # 5. Convert similarity scores to probability distributions.
        student_probs_log = F.log_softmax(student_sim, dim=1)
        teacher_probs = F.softmax(teacher_sim, dim=1)
        
        # 6. Compute KL divergence loss (using "batchmean" reduction).
        kl_loss = F.kl_div(student_probs_log, teacher_probs, reduction='batchmean')
        
        # 7. Return the weighted sum of the base loss and the KL loss.
        return self.alpha * base_loss_value + (1 - self.alpha) * kl_loss

# ---------------------------
# SBERTTuner Class
# ---------------------------
class SBERTTuner:
    def __init__(self, dataset, dataset_name, model_name='all-MiniLM-L6-v2', seed=42, loss_function=None):
        """
        Initialize the tuner with a dataset instance.
        The dataset is expected to implement get_splits(), returning:
            ((train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels))
        """
        self.seed = seed
        self.model_name = model_name
        self.dataset = dataset
        self.set_seed(seed)
        
        # Parse splits from the dataset.
        (self.train_texts, self.train_labels), (self.val_texts, self.val_labels), (self.test_texts, self.test_labels) = dataset.get_splits()
        self.dataset_name = dataset_name
        
        # Initialize the student model.
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                     else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
        
        # Use MultipleNegativesRankingLoss by default if no loss function is provided.
        self.loss_function = losses.MultipleNegativesRankingLoss(self.model) if loss_function is None else loss_function(self.model)
    
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def create_training_pairs(self, texts, labels, num_pairs=10000):
        """
        Create training pairs given texts and labels.
        For each training pair, two sentences from the same label are sampled.
        """
        label_to_texts = {}
        for text, label in zip(texts, labels):
            label_to_texts.setdefault(label, []).append(text)
            
        training_pairs = []
        unique_labels = list(label_to_texts.keys())
        
        for _ in range(num_pairs):
            pos_label = random.choice(unique_labels)
            if len(label_to_texts[pos_label]) < 2:
                continue
            pair = random.sample(label_to_texts[pos_label], 2)
            training_pairs.append(InputExample(texts=pair))
        
        random.shuffle(training_pairs)
        return training_pairs

    def fine_tune(self, batch_size=16, epochs=1, warmup_steps=100, learning_rate=2e-5,
                  weight_decay=0.01, max_grad_norm=0.5, output_parent_path='models/tuned_sbert/',
                  use_student_teacher=False, kl_alpha=0.7, temperature=2.0):
        """
        Fine-tune the SBERT model using the default training loop.
        Optionally, a student-teacher setup with KL divergence is applied to prevent catastrophic forgetting.
        
        Parameters:
          - use_student_teacher (bool): If True, use the student-teacher loss.
          - kl_alpha (float): Weight factor for the base loss (the remaining weight is given to the KL divergence term).
          - temperature (float): Temperature for smoothing the distributions used in KL divergence.
        """
        # Create training pairs.
        train_examples = self.create_training_pairs(self.train_texts, self.train_labels)
        train_dataset = SentencesDataset(train_examples, model=self.model)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=self.model.smart_batching_collate)
        
        # Optionally set up the student-teacher configuration.
        if use_student_teacher:
            # Create a teacher model (a frozen copy of the original model).
            teacher_model = SentenceTransformer(self.model_name)
            teacher_model.to(self.device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            # Wrap the base loss with the StudentTeacherLoss.
            loss_to_use = StudentTeacherLoss(self.model, teacher_model, self.loss_function,
                                             temperature=temperature, alpha=kl_alpha)
            print("Using Student-Teacher KL Divergence Loss with temperature={} and kl_alpha={}".format(temperature, kl_alpha))
        else:
            loss_to_use = self.loss_function
            print("Using base loss only: {}".format(loss_to_use.__class__.__name__))
        
        # Define a callback function to be called at the end of each epoch.
        def epoch_callback(avg_loss, epoch, steps):
            print(f"Epoch {epoch} ended. Average Loss = {avg_loss:.4f}")
        
        output_path = os.path.join(output_parent_path, self.dataset_name)
        
        # Run the training loop using the model's built-in fit() method.
        self.model.fit(
            train_objectives=[(train_dataloader, loss_to_use)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            optimizer_params={'lr': learning_rate},  # Lower this value if needed
            max_grad_norm=max_grad_norm,
            output_path=output_path,
            callback=epoch_callback
        )
        
        print(f"Model saved at {output_path}")



