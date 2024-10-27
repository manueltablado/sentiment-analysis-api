import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch

# Load and preprocess the dataset
def prepare_data(filepath="data/reviews.tsv"):
    df = pd.read_csv(filepath, sep='\t')
    df.dropna(subset=['Review', 'Liked'], inplace=True)
    df['Liked'] = df['Liked'].apply(lambda x: 1 if x == 1 else 0)
    
    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Liked'], random_state=42)
    
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize the reviews
    train_encodings = tokenizer(list(train_df['Review']), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_df['Review']), truncation=True, padding=True, max_length=128)
    
    # Prepare datasets for PyTorch
    class ReviewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_labels = list(train_df['Liked'])
    val_labels = list(val_df['Liked'])

    train_dataset = ReviewsDataset(train_encodings, train_labels)
    val_dataset = ReviewsDataset(val_encodings, val_labels)

    return train_dataset, val_dataset, val_labels
