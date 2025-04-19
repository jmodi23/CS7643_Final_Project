# import os
# import random
# import numpy as np
# import itertools
# import json
# import torch
# from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class SBERTTuner:
#     def __init__(self, dataset, dataset_name, model_name='all-MiniLM-L6-v2', seed=42, loss_function=None):
#         """
#         Initialize the tuner with a dataset instance.
#         The dataset is expected to implement get_splits(), returning:
#             ((train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels))
#         """
#         self.seed = seed
#         self.model_name = model_name
#         self.dataset = dataset
#         self.set_seed(seed)
        
#         # Parse splits from the dataset.
#         (self.train_texts, self.train_labels), (self.val_texts, self.val_labels), (self.test_texts, self.test_labels) = dataset.get_splits()
#         self.dataset_name = dataset_name
        
#         # Initialize the model.
#         self.device = torch.device("mps" if torch.backends.mps.is_available() 
#                                      else "cuda" if torch.cuda.is_available() else "cpu")
#         self.model = SentenceTransformer(self.model_name)
#         self.model.to(self.device)
        
#         # Use MultipleNegativesRankingLoss by default if no loss function is provided.
#         self.loss_function = losses.MultipleNegativesRankingLoss(self.model) if loss_function is None else loss_function(self.model)
    
#     def set_seed(self, seed):
#         np.random.seed(seed)
#         random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
    
#     def create_training_pairs(self, texts, labels, num_pairs=10000):
#         """
#         Create training pairs given texts and labels.
#         For each training pair, two sentences from the same label are sampled.
#         """
#         label_to_texts = {}
#         for text, label in zip(texts, labels):
#             label_to_texts.setdefault(label, []).append(text)
            
#         training_pairs = []
#         unique_labels = list(label_to_texts.keys())
        
#         for _ in range(num_pairs):
#             pos_label = random.choice(unique_labels)
#             if len(label_to_texts[pos_label]) < 2:
#                 continue
#             pair = random.sample(label_to_texts[pos_label], 2)
#             training_pairs.append(InputExample(texts=pair))
        
#         random.shuffle(training_pairs)
#         return training_pairs

#     def fine_tune(self, batch_size=16, epochs=1, warmup_steps=100, learning_rate=2e-5,
#                   weight_decay=0.01, max_grad_norm=0.5, output_parent_path='models/tuned_sbert/'):
#         """
#         Fine-tune the SBERT model using the default training loop.
#         A custom callback prints the average final loss at the end of each epoch.
#         """
#         # Create training pairs using the new function.
#         train_examples = self.create_training_pairs(self.train_texts, self.train_labels)
#         train_dataset = SentencesDataset(train_examples, model=self.model)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                       collate_fn=self.model.smart_batching_collate)
        
#         # Define a callback function to be called at the end of each epoch.
#         def epoch_callback(avg_loss, epoch, steps):
#             print(f"Epoch {epoch} ended. Average Loss = {avg_loss:.4f}")
        
#         output_path = os.path.join(output_parent_path, self.dataset_name)
        
#         # Use the default training loop with the built-in fit method.
#         self.model.fit(
#             train_objectives=[(train_dataloader, self.loss_function)],
#             epochs=epochs,
#             warmup_steps=warmup_steps,
#             weight_decay=weight_decay,
#             optimizer_params={'lr': learning_rate},
#             max_grad_norm=max_grad_norm,
#             output_path=output_path,
#             # The callback function is called at the end of each epoch.
#             callback=epoch_callback
#         )
        
#         print(f"Model saved at {output_path}")

# Example usage:
# Make sure your dataset implements get_splits()
# dataset = YourDatasetImplementation(...)
# tuner = SBERTTuner(dataset, dataset_name="20Newsgroups")
# tuner.fine_tune(batch_size=8, epochs=3, warmup_steps=100, learning_rate=1e-05, weight_decay=0, max_grad_norm=0.5)

# Example usage:
# Assuming you have a dataset implementation that provides get_splits()
# dataset = YourDatasetImplementation(...)
# tuner = SBERTTuner(dataset, dataset_name="20Newsgroups")
# tuner.fine_tune(batch_size=8, epochs=1, warmup_steps=100, learning_rate=1e-05, weight_decay=0, max_grad_norm=0.5)

# Example usage:
# Assuming you have a dataset that implements get_splits():
# dataset = YourDatasetImplementation(...)
# tuner = SBERTTuner(dataset, dataset_name="20Newsgroups")
# tuner.fine_tune(batch_size=8, epochs=1, learning_rate=1e-05, weight_decay=0, max_grad_norm=0.5)
# Example usage (assuming you have a dataset that implements get_splits()):
# dataset = YourDatasetImplementation(...)
# tuner = SBERTTuner(dataset, dataset_name="20Newsgroups")
# tuner.fine_tune(batch_size=8, epochs=1, learning_rate=1e-05, weight_decay=0, max_grad_norm=0.5)

# Example usage (assuming you have a dataset that implements get_splits()):
# dataset = YourDatasetImplementation(...)
# tuner = SBERTTuner(dataset, dataset_name="20Newsgroups")
# tuner.fine_tune(batch_size=8, epochs=1, learning_rate=1e-05, weight_decay=0, max_grad_norm=0.5)


# import os
# import random
# import numpy as np
# import itertools
# import json
# import torch
# from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset
# from sentence_transformers.evaluation import SentenceEvaluator
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class SBERTTuner:
#     def __init__(self, dataset, dataset_name, model_name='all-MiniLM-L6-v2', seed=42, loss_function=None):
#         """
#         Initialize the tuner with a dataset instance.
#         The dataset is expected to implement get_splits(), returning:
#             ((train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels))
#         """
#         self.seed = seed
#         self.model_name = model_name
#         self.dataset = dataset
#         self.set_seed(seed)
        
#         # Parse splits from the dataset.
#         (self.train_texts, self.train_labels), (self.val_texts, self.val_labels), (self.test_texts, self.test_labels) = dataset.get_splits()
#         # Optionally, store the dataset name if available.
#         self.dataset_name = dataset_name
        
#         # Initialize the model.
#         self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#         self.model = SentenceTransformer(self.model_name)
#         self.model.to(self.device)
        
#         # Use MultipleNegativesRankingLoss by default if no loss function is provided.
#         self.loss_function = losses.MultipleNegativesRankingLoss(self.model) if loss_function is None else loss_function(self.model)
    
#     def set_seed(self, seed):
#         np.random.seed(seed)
#         random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
    
#     def create_training_data(self, texts, labels, num_samples=10000, n_candidates=5):
#         """
#         Create training examples as InputExample instances with hard negative mining.
#         Positive pairs get label=1 and negative pairs get label=0.
        
#         For each anchor, for negative examples, we sample a few candidates from texts with a different label
#         and select the one that is most similar (i.e., "hardest negative") using TF-IDF cosine similarity.
#         """
#         label_to_texts = {}
#         for text, label in zip(texts, labels):
#             label_to_texts.setdefault(label, []).append(text)

#         train_examples = []
#         unique_labels = list(label_to_texts.keys())

#         # Build a TF-IDF vectorizer on all training texts.
#         vectorizer = TfidfVectorizer().fit(texts)
#         # Precompute TF-IDF vectors for all texts.
#         text_vectors = {text: vectorizer.transform([text]) for text in texts}

#         # Shuffle texts within each label.
#         for label in unique_labels:
#             random.shuffle(label_to_texts[label])

#         for _ in range(num_samples // 2):
#             # Select a positive label.
#             pos_label = random.choice(unique_labels)
#             if len(label_to_texts[pos_label]) < 2:
#                 continue
#             # Randomly sample a positive pair.
#             pos_pair = random.sample(label_to_texts[pos_label], 2)
#             train_examples.append(InputExample(texts=pos_pair, label=1))
            
#             # For a hard negative: use the first sentence of the positive pair as the anchor.
#             anchor = pos_pair[0]
            
#             # Build candidate negatives: texts from all labels except pos_label.
#             non_pos_texts = [t for l in unique_labels if l != pos_label for t in label_to_texts[l]]
#             if len(non_pos_texts) == 0:
#                 continue  # Should not happen unless there is only one label.
#             # Sample candidates.
#             if len(non_pos_texts) < n_candidates:
#                 candidates = non_pos_texts
#             else:
#                 candidates = random.sample(non_pos_texts, n_candidates)
            
#             # Compute cosine similarity between the anchor and each candidate.
#             anchor_vec = text_vectors[anchor]
#             candidate_scores = [cosine_similarity(anchor_vec, text_vectors[candidate])[0, 0] for candidate in candidates]
#             # Choose the candidate with the highest similarity.
#             hard_negative = candidates[np.argmax(candidate_scores)]
#             neg_pair = [anchor, hard_negative]
#             train_examples.append(InputExample(texts=neg_pair, label=0))
        
#         random.shuffle(train_examples)
#         return train_examples[:num_samples]


#     def create_pair_validation_data(self, texts, labels, num_samples=1000, n_candidates=5):
#         """
#         Create a validation set for pair classification with a mix of positive and hard negative pairs.
        
#         For positive pairs, all combinations (or a subset) are considered.
#         For negative pairs, for each anchor, n_candidates are sampled from different labels,
#         and the "hardest" negative (with highest cosine similarity using TF-IDF) is chosen.
#         """
#         label_to_texts = {}
#         for text, label in zip(texts, labels):
#             label_to_texts.setdefault(label, []).append(text)

#         pos_examples = []
#         neg_examples = []
#         unique_labels = list(label_to_texts.keys())

#         # Build TF-IDF vectorizer and precompute vectors.
#         vectorizer = TfidfVectorizer().fit(texts)
#         text_vectors = {text: vectorizer.transform([text]) for text in texts}

#         # Create all possible positive pairs for each label.
#         for label in unique_labels:
#             texts_for_label = label_to_texts[label]
#             if len(texts_for_label) < 2:
#                 continue
#             pairs = list(itertools.combinations(texts_for_label, 2))
#             pos_examples.extend([InputExample(texts=list(pair), label=1) for pair in pairs])
        
#         # For negative pairs, for each positive pair in pos_examples, build a negative pair.
#         for example in pos_examples:
#             anchor = example.texts[0]
#             # Build candidate negatives from texts with a different label.
#             non_pos_texts = [t for l in unique_labels if l != self.dataset_name for t in label_to_texts.get(l, [])]
#             # Alternatively, if you want to sample uniformly from all texts not in the positive pair:
#             non_pos_texts = [t for l in unique_labels for t in label_to_texts[l] if t not in example.texts]
#             if len(non_pos_texts) == 0:
#                 continue
#             if len(non_pos_texts) < n_candidates:
#                 candidates = non_pos_texts
#             else:
#                 candidates = random.sample(non_pos_texts, n_candidates)
#             anchor_vec = text_vectors[anchor]
#             candidate_scores = [cosine_similarity(anchor_vec, text_vectors[candidate])[0, 0] for candidate in candidates]
#             hard_negative = candidates[np.argmax(candidate_scores)]
#             neg_examples.append(InputExample(texts=[anchor, hard_negative], label=0))
        
#         all_examples = pos_examples + neg_examples
#         random.shuffle(all_examples)
#         if len(all_examples) > num_samples:
#             all_examples = random.sample(all_examples, num_samples)
#         return all_examples

#     def fine_tune(self, batch_size=16, epochs=1, warmup_steps=100, weight_decay=0.01, learning_rate=2e-5, max_grad_norm=0.5, evaluation_steps=100, output_parent_path='models/tuned_sbert/'):
#         """
#         Fine-tune the SBERT model using training and validation sets.
#         The evaluation runs during training and monitors performance on the validation set.
#         """
#         # Create training and validation InputExamples.
#         train_examples = self.create_training_data(self.train_texts, self.train_labels)
#         val_examples = self.create_pair_validation_data(self.val_texts, self.val_labels)
        
#         # Build the training dataset.
#         train_dataset = SentencesDataset(train_examples, model=self.model)
#         # Use the model's smart batching collate function for training.
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.model.smart_batching_collate)
        
#         # Build the validation dataset for evaluation.
#         # Here we use an EmbeddingSimilarityEvaluator, but because our validation set is binary,
#         # it is better to use a custom evaluator for classification. For illustration, we
#         # compute similarity scores and then threshold them.
#         # We extract pairs and labels:
#         val_pairs = [ex.texts for ex in val_examples]
#         val_labels = [ex.label for ex in val_examples]
        
#         evaluator = PairwiseClassificationEvaluator(val_pairs, val_labels, batch_size=batch_size, threshold=0.5, name="val-classifier")
        
#         # Fine-tune the model using model.fit(), which supports an evaluator.
#         self.model.fit(
#             train_objectives=[(train_dataloader, self.loss_function)],
#             evaluator=evaluator,
#             epochs=epochs,
#             warmup_steps=warmup_steps,
#             weight_decay=weight_decay,
#             optimizer_params={'lr': learning_rate},
#             max_grad_norm=max_grad_norm,
#             evaluation_steps=evaluation_steps
#         )
        
#         os.makedirs(output_parent_path, exist_ok=True)
#         output_path = os.path.join(output_parent_path, self.dataset_name)
#         self.model.save(output_path)
#         print(f"Model saved at {output_path}")
    
#     def hyperparameter_tuning(self, param_grid=None, base_output_path='models/tuned_sbert/', k=5):
#         """
#         Performs hyperparameter tuning by iterating through a grid of parameters.
#         For each combination, a new SBERT model is fine-tuned and evaluated via a FAISS-based
#         kNN classifier and F1-score.
#         """
#         if param_grid is None:
#             param_grid = {
#                 'batch_size': [8, 16],
#                 'epochs': [1, 2],
#                 'warmup_steps': [50, 100],
#                 'learning_rate': [1e-5, 2e-5],
#                 'weight_decay': [0.0, 0.01],
#                 'max_grad_norm': [0.5, 1.0]
#             }
        
#         grid_keys = list(param_grid.keys())
#         grid_values = [param_grid[k] for k in grid_keys]
#         param_combinations = list(itertools.product(*grid_values))
        
#         best_score = -float("inf")
#         best_params = None
#         results = []
        
#         for idx, values in enumerate(param_combinations):
#             params = dict(zip(grid_keys, values))
#             print(f"Run {idx+1}/{len(param_combinations)} with parameters: {params}")
#             run_output_path = os.path.join(base_output_path, self.dataset_name, f"run_{idx+1}")
#             os.makedirs(run_output_path, exist_ok=True)
            
#             new_tuner = type(self)(dataset=self.dataset, model_name=self.model_name, seed=self.seed)
#             new_tuner.fine_tune(
#                 batch_size=params['batch_size'],
#                 epochs=params['epochs'],
#                 warmup_steps=params['warmup_steps'],
#                 weight_decay=params['weight_decay'],
#                 learning_rate=params['learning_rate'],
#                 max_grad_norm=params['max_grad_norm'],
#                 output_parent_path=run_output_path
#             )
            
#             # --- Evaluation via FAISS or other method ---
#             # For demonstration we assume a dummy evaluation (for a real case, implement FAISS-based kNN).
#             val_pairs = [ex.texts for ex in new_tuner.create_pair_validation_data(self.val_texts, self.val_labels)]
#             # For now, we use majority class prediction as a placeholder.
#             preds = [max(set(self.train_labels), key=self.train_labels.tolist().count)] * len(self.val_texts)
#             score_value = f1_score(self.val_labels, preds, average='weighted')
#             print(f"  Validation F1-score: {score_value:.4f}\n")
            
#             run_result = params.copy()
#             run_result.update({'run_idx': idx+1, 'score': score_value})
#             results.append(run_result)
            
#             if score_value > best_score:
#                 best_score = score_value
#                 best_params = run_result
        
#         print("=======================================")
#         print("Best hyperparameters found:")
#         print(best_params)
#         print(f"with F1-score: {best_score:.4f}")
#         results_file = os.path.join(base_output_path, self.dataset_name, "hyperparameter_tuning_results.json")
#         with open(results_file, "w") as f:
#             json.dump(results, f, indent=4)
#         print(f"All tuning results saved to {results_file}")
        
#         return best_params, results

# # ------------------------------
# # Custom Evaluator for Binary Pair Classification
# # ------------------------------

# class PairwiseClassificationEvaluator(SentenceEvaluator):
#     """
#     Evaluates the model by encoding pairs from validation data, computing cosine similarities,
#     thresholding those similarities, and then calculating classification metrics (F1 and accuracy).
#     """
#     def __init__(self, pairs, labels, batch_size=32, threshold=0.5, name="PairwiseClassifier"):
#         """
#         Args:
#             pairs (list): List of pairs, where each pair is a list [sentence0, sentence1].
#             labels (list): List of ground truth labels (0 for dissimilar, 1 for similar).
#             batch_size (int): Batch size for encoding.
#             threshold (float): Cosine similarity threshold to decide positive vs. negative.
#             name (str): Name of the evaluator.
#         """
#         self.pairs = pairs
#         self.labels = labels
#         self.batch_size = batch_size
#         self.threshold = threshold
#         self.name = name

#     def __call__(self, model, output_path=None, epoch=-1, steps=-1):
#         model.eval()
#         cosine_scores = []
#         # Process in batches
#         for start in range(0, len(self.pairs), self.batch_size):
#             batch = self.pairs[start: start+self.batch_size]
#             sents1 = [pair[0] for pair in batch]
#             sents2 = [pair[1] for pair in batch]
#             emb1 = model.encode(sents1, batch_size=self.batch_size, convert_to_tensor=True)
#             emb2 = model.encode(sents2, batch_size=self.batch_size, convert_to_tensor=True)
#             # Normalize and compute cosine similarity
#             sim = (torch.nn.functional.normalize(emb1, p=2, dim=1) * torch.nn.functional.normalize(emb2, p=2, dim=1)).sum(dim=1)
#             cosine_scores.extend(sim.cpu().tolist())
#         # Apply threshold to decide positive vs. negative.
#         predictions = [1 if score >= self.threshold else 0 for score in cosine_scores]
#         f1 = f1_score(self.labels, predictions, average='weighted')
#         acc = accuracy_score(self.labels, predictions)
#         print(f"{self.name} (Epoch {epoch}, Steps {steps}): F1={f1:.4f}, Accuracy={acc:.4f}")
#         return f1

#     def to_dict(self):
#         return {"name": self.name, "batch_size": self.batch_size, "threshold": self.threshold}


# import torch
# from sentence_transformers import SentenceTransformer, losses, InputExample, SentencesDataset
# from torch.utils.data import DataLoader
# import os
# import random
# import numpy as np
# import itertools
# from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
# from sklearn.metrics import f1_score
# from .semantic_search import FaissIndex
# import json



# class SBERTTuner:
#     def __init__(self, dataset_name, model_name='all-MiniLM-L6-v2', seed=42, loss_function=None):
#         self.seed = seed
#         self.model_name = model_name
#         self.dataset_name = dataset_name
#         self.set_seed(seed)
        
#         self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#         self.model = SentenceTransformer(self.model_name)
#         self.model.to(self.device)
#         # Use MultipleNegativesRankingLoss by default if no loss function is provided
#         self.loss_function = losses.MultipleNegativesRankingLoss(self.model) if loss_function is None else loss_function(self.model)
    
#     def set_seed(self, seed):
#         np.random.seed(seed)
#         random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(seed)
    
#     def create_training_data(self, texts, labels, num_samples=10000):
#         label_to_texts = {}
#         for text, label in zip(texts, labels):
#             label_to_texts.setdefault(label, []).append(text)

#         train_examples = []
#         unique_labels = list(label_to_texts.keys())
#         for label in unique_labels:
#             random.shuffle(label_to_texts[label])

#         for _ in range(num_samples // 2):
#             pos_label = random.choice(unique_labels)
#             if len(label_to_texts[pos_label]) < 2:
#                 continue
#             pos_pair = random.sample(label_to_texts[pos_label], 2)
#             train_examples.append(InputExample(texts=pos_pair, label=1))

#             neg_label = random.choice([l for l in unique_labels if l != pos_label])
#             neg_sample = random.choice(label_to_texts[neg_label])
#             neg_pair = [pos_pair[0], neg_sample]
#             train_examples.append(InputExample(texts=neg_pair, label=0))

#         random.shuffle(train_examples)
#         return train_examples[:num_samples]
    
#     def create_validation_data(self, texts, labels, num_samples=1000):
#         # Create validation pairs similar to training
#         label_to_texts = {}
#         for text, label in zip(texts, labels):
#             label_to_texts.setdefault(label, []).append(text)

#         val_examples = []
#         unique_labels = list(label_to_texts.keys())
#         for label in unique_labels:
#             texts_for_label = label_to_texts[label]
#             if len(texts_for_label) < 2:
#                 continue
#             # Create all possible pairs for the label
#             pairs = list(itertools.combinations(texts_for_label, 2))
#             # If there are too many pairs, sample a subset
#             if len(pairs) > num_samples // len(unique_labels):
#                 pairs = random.sample(pairs, num_samples // len(unique_labels))
#             for pair in pairs:
#                 val_examples.append(InputExample(texts=list(pair), label=1))
#         return val_examples
    
#     def fine_tune(self, train_texts, train_labels, batch_size=16, epochs=1, warmup_steps=100, 
#                   weight_decay=0.01, learning_rate=2e-5, max_grad_norm=0.5, output_parent_path='models/tuned_sbert/'):
#         train_examples = self.create_training_data(train_texts, train_labels)
#         train_dataset = SentencesDataset(train_examples, model=self.model)
#         # Use the model's smart batching collate function to create batches as (features, labels)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
#                                       collate_fn=self.model.smart_batching_collate)
        
#         self.model.fit(
#             train_objectives=[(train_dataloader, self.loss_function)],
#             epochs=epochs,
#             warmup_steps=warmup_steps,
#             weight_decay=weight_decay,
#             optimizer_params={'lr': learning_rate},
#             max_grad_norm=max_grad_norm
#         )
        
#         os.makedirs(output_parent_path, exist_ok=True)
#         output_path = os.path.join(output_parent_path, self.dataset_name)
#         self.model.save(output_path)
#         print(f"Model saved at {output_path}")
        
        
#     def hyperparameter_tuning(self, train_texts, train_labels, val_texts, val_labels,
#                             param_grid=None, num_train_samples=10000, num_val_samples=1000,
#                             base_output_path='models/tuned_sbert/', k=5):
#         """
#         Performs hyperparameter tuning by iterating through a grid of parameters.
#         For each combination, a new SBERT model is fine-tuned and evaluated on
#         validation data using a FAISS-based kNN classifier and F1-score as the metric.
        
#         Args:
#             train_texts (list): Training texts.
#             train_labels (list or np.array): Training labels.
#             val_texts (list): Validation texts.
#             val_labels (list or np.array): Validation labels.
#             param_grid (dict, optional): Dictionary with hyperparameter lists.
#                                         Defaults to preset values if None.
#             num_train_samples (int): (Not used here; kept for consistency) Number
#                                     of training samples to use.
#             num_val_samples (int): (Not used here; kept for consistency) Number of validation samples.
#             base_output_path (str): Parent directory for saving fine-tuned models.
#             k (int): Number of nearest neighbors for FAISS classification.
        
#         Returns:
#             tuple: (best_params, results) where best_params is the dict with the best
#                 hyperparameters and results is a list containing all run results.
#         """

#         # Set default hyperparameter grid if none is provided.
#         if param_grid is None:
#             param_grid = {
#                 'batch_size': [8, 16],
#                 'epochs': [1, 2],
#                 'warmup_steps': [50, 100],
#                 'learning_rate': [1e-5, 2e-5],
#                 'weight_decay': [0.0, 0.01],
#                 'max_grad_norm': [0.5, 1.0]
#             }
        
#         grid_keys = list(param_grid.keys())
#         grid_values = [param_grid[k] for k in grid_keys]
#         param_combinations = list(itertools.product(*grid_values))
        
#         best_score = -float("inf")
#         best_params = None
#         results = []

#         # Iterate over each hyperparameter combination.
#         for idx, values in enumerate(param_combinations):
#             params = dict(zip(grid_keys, values))
#             print(f"Run {idx+1}/{len(param_combinations)} with parameters: {params}")
            
#             run_output_path = os.path.join(base_output_path, self.dataset_name, f"run_{idx+1}")
#             os.makedirs(run_output_path, exist_ok=True)
            
#             # Create a new tuner instance so each run starts from the same pre-trained weights.
#             new_tuner = type(self)(dataset_name=self.dataset_name, model_name=self.model_name, seed=self.seed)
            
#             # Fine-tune the model using the current hyperparameters.
#             new_tuner.fine_tune(
#                 train_texts=train_texts,
#                 train_labels=train_labels,
#                 batch_size=params['batch_size'],
#                 epochs=params['epochs'],
#                 warmup_steps=params['warmup_steps'],
#                 weight_decay=params['weight_decay'],
#                 learning_rate=params['learning_rate'],
#                 max_grad_norm=params['max_grad_norm'],
#                 output_parent_path=run_output_path
#             )
            
#             # --- Classification Evaluation via FAISS ---
#             # Build FAISS index on training embeddings.
#             train_embeddings = new_tuner.model.encode(train_texts)
#             train_labels_np = np.array(train_labels)
#             # Assume FaissIndex is defined and imported.
#             faiss_index = FaissIndex(train_embeddings, train_labels_np)
            
#             # Generate predictions on the validation set.
#             val_embeddings = new_tuner.model.encode(val_texts)
#             preds = []
#             for i in range(len(val_texts)):
#                 query_emb = val_embeddings[i].reshape(1, -1)
#                 pred = faiss_index.classify(query_emb, k=k)
#                 preds.append(pred)
            
#             # Compute weighted F1-score.
#             score_value = f1_score(val_labels, preds, average='weighted')
#             print(f"  Validation F1-score: {score_value:.4f}\n")
            
#             run_result = params.copy()
#             run_result.update({'run_idx': idx+1, 'score': score_value})
#             results.append(run_result)
            
#             if score_value > best_score:
#                 best_score = score_value
#                 best_params = run_result

#         print("=======================================")
#         print("Best hyperparameters found:")
#         print(best_params)
#         print(f"with F1-score: {best_score:.4f}")
        
#         # Save all tuning results to a JSON file.
#         results_file = os.path.join(base_output_path, self.dataset_name, "hyperparameter_tuning_results.json")
#         with open(results_file, "w") as f:
#             json.dump(results, f, indent=4)
#         print(f"All tuning results saved to {results_file}")
        
#         return best_params, results