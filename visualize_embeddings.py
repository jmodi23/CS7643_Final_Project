import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
import torch

from data.amazonreviews_data import AmazonReviewsDataset
from data.newsgroups_data import NewsGroupsDataset
from data.trec_data import TrecDataset
from modules.embedding_model import EmbeddingModel

def get_device():
    """Get the appropriate device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def visualize_embeddings(embeddings, labels, label_names, title, ax=None):
    """Create t-SNE visualization of embeddings"""
    print(f"Running t-SNE for {title}...")
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': [label_names[label] for label in labels]
    })
    
    num_classes = len(np.unique(labels))
    palette = sns.color_palette("colorblind", n_colors=num_classes)
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    sns.scatterplot(
        data=df, x='x', y='y', hue='label', 
        palette=palette, alpha=0.7, s=50, ax=ax
    )
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE dimension 2', fontsize=10)
    ax.legend(title='Class', loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0, fontsize=8)
    ax.grid(False)#, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return ax

def get_embeddings(dataset, model_name, sample_size=2000):
    """Get embeddings for dataset using specified model"""
    (train_texts, train_labels), (_, _), (_, _) = dataset.get_splits()
    label_names = dataset.label_names
    
    print(f"Encoding texts with {model_name}...")
    embedding_model = EmbeddingModel(model_name)
    
    # Sample if dataset is large
    if len(train_texts) > sample_size:
        indices = np.random.choice(len(train_texts), sample_size, replace=False)
        sample_texts = [train_texts[i] for i in indices]
        sample_labels = train_labels[indices]
    else:
        sample_texts = train_texts
        sample_labels = train_labels
    
    # Get embeddings
    sample_embeddings = embedding_model.encode_texts(sample_texts)
    
    return sample_embeddings, sample_labels, label_names

def visualize_dataset_with_model(dataset_name, model_name, finetuned_model_path=None):
    """Create 2×1 visualization for a single dataset with a specific model, comparing before and after fine-tuning"""
    # Get dataset display name
    if dataset_name == 'amazon':
        display_name = 'Amazon Reviews'
    elif dataset_name == 'newsgroups':
        display_name = 'News Groups'
    elif dataset_name == 'trec':
        display_name = 'TREC Questions'
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Get model display name
    if 'MiniLM' in model_name:
        model_display_name = 'SBERT (MiniLM)'
    elif 'distilbert' in model_name.lower():
        model_display_name = 'DistilBERT'
    else:
        model_display_name = model_name
    
    # Create a 2×1 figure (2 rows for before/after, 1 column for the dataset)
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    #fig.suptitle(f'Embedding Visualizations: {display_name} Dataset with {model_display_name}', 
    #             fontsize=16, fontweight='bold', y=0.98)
    
    # Row titles
    # axes[0].set_title(f'Before Fine-tuning {display_name}', fontsize=14, fontweight='bold')
    # if finetuned_model_path:
    #     axes[1].set_title(f'After Fine-tuning {display_name}', fontsize=14, fontweight='bold')
    # else:
    #     axes[1].set_title('After Fine-tuning (Placeholder)', fontsize=14, fontweight='bold')
    
    print(f"\nProcessing {display_name} dataset with {model_display_name}...")
    
    # Get dataset
    dataset = get_dataset(dataset_name)
    label_names = dataset.label_names

    # Get embeddings
    embeddings_before, labels, _ = get_embeddings(dataset, model_name)
    
    # Create "before" visualization (top row)
    visualize_embeddings(embeddings_before, labels, label_names, 
                         f'Before Fine-tuning {display_name}', ax=axes[0])
    
    if finetuned_model_path is not None:
        print(f"Computing fine-tuned embeddings for {dataset_name}...")
        embeddings_after, labels, _ = get_embeddings(dataset, finetuned_model_path)
        
        visualize_embeddings(embeddings_after, labels, label_names,
                             f'After Fine-tuning {display_name}', ax=axes[1])
    else:
        # Placeholder
        axes[1].text(0.5, 0.5, "Fine-tuned embeddings\nwill appear here", 
                   ha='center', va='center', fontsize=14, 
                   transform=axes[1].transAxes)
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    model_filename = 'sbert' if 'MiniLM' in model_name else 'distilbert'
    
    filename = f'images/embeddings_{dataset_name}_{model_filename}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to '{filename}'")
    
    plt.close(fig)

def process_dataset_with_multiple_models(dataset_name, models=None, finetuned_models=None):
    """Process a dataset with multiple embedding models"""
    # Default models
    if models is None:
        models = ['all-MiniLM-L6-v2', 'distilbert-base-uncased']
    
    # Default finetuned models
    if finetuned_models is None:
        finetuned_models = {model: None for model in models}
    
    for model in models:
        visualize_dataset_with_model(
            dataset_name, 
            model, 
            finetuned_models.get(model)
        )

# EDIT THIS
def get_dataset(dataset_name):
    """Get dataset instance based on name"""
    if dataset_name == 'amazon':
        params = {
            'csv_path': 'eda_outputs/amazon_reviews_subset_20k.csv',
            'holdout_classes': ['Books']
        }
        dataset = AmazonReviewsDataset(**params)
    elif dataset_name == 'newsgroups':
        params = {
            'categories': ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.mideast'],
            'holdout_classes': ['talk.politics.mideast']
        }
        dataset = NewsGroupsDataset(**params)
    elif dataset_name == 'trec':
        params = {
            'holdout_classes': ['ENTY:currency', 'ENTY:religion', 'NUM:ord', 'NUM:temp', 'ENTY:letter', 'NUM:code', 'NUM:speed',
            'ENTY:instru', 'ENTY:symbol', 'NUM:weight', 'ENTY:plant', 'NUM:volsize', 'ABBR:abb', 'ENTY:body', 'ENTY:lang', 'LOC:mount',
            'HUM:title', 'ENTY:word','ENTY:veh', 'NUM:perc', 'NUM:dist', 'ENTY:techmeth', 'ENTY:color', 'ENTY:substance','ENTY:product',
            'HUM:desc',]
        }
        dataset = TrecDataset(**params)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset

if __name__ == "__main__":
    np.random.seed(42)
    
    import os
    os.makedirs("images", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    
    # Options: 'amazon', 'newsgroups', 'trec'
    dataset_name = 'amazon'  # CHANGE
    
    # Models to use
    models = ['all-MiniLM-L6-v2', 'distilbert-base-uncased']
    # process_dataset_with_multiple_models(dataset_name, models)
    
    # EDIT THE BELOW PATHS:
    # finetuned_models = {
    #     'all-MiniLM-L6-v2': 'path/to/finetuned/sbert/model',
    #     'distilbert-base-uncased': 'path/to/finetuned/distilbert/model'
    # }
    finetuned_models = {
        'all-MiniLM-L6-v2': 'all-MiniLM-L6-v2',
        'distilbert-base-uncased': 'distilbert-base-uncased'
    }
    process_dataset_with_multiple_models(dataset_name, models, finetuned_models)