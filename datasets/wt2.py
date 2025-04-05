import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from pathlib import Path

class WT2Dataset:
    def __init__(self, seq_length=35, batch_size=20):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_dir = Path("datasets/wikitext2")
        self.tokenizer = None
        self.vocab_size = 0
        
        # Create datasets directory if it doesn't exist
        Path("datasets").mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data using HuggingFace datasets with local caching
        self.dataset = self.load_wikitext_dataset()
        
        # Split data
        self.train_texts = self.dataset["train"]["text"]
        self.val_texts = self.dataset["validation"]["text"]
        self.test_texts = self.dataset["test"]["text"]
        
        # Process the data
        self.x_train, self.y_train = self.process_data(self.train_texts)
        self.x_val, self.y_val = self.process_data(self.val_texts)
        self.x_test, self.y_test = self.process_data(self.test_texts)
        
        # Set input and output shapes for model building
        self.input_shape = (self.seq_length,)
        self.output_dim = self.vocab_size
    
    def load_wikitext_dataset(self):
        """Load WikiText-2 dataset with local caching"""
        # Set the cache directory to our specific location
        cache_dir = str(self.data_dir)
        
        print(f"Loading WikiText-2 dataset (cached at {cache_dir})...")
        try:
            # Try to load the dataset with local caching
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)
            print("WikiText-2 dataset loaded successfully.")
            return dataset
        except Exception as e:
            print(f"Error loading WikiText-2 dataset: {e}")
            raise
    
    def process_data(self, texts):
        """Process text data into training samples"""
        # Filter out empty lines
        filtered_texts = [text for text in texts if text.strip()]
        
        # Join all texts
        combined_text = " ".join(filtered_texts)
        
        # Tokenize on first process only
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(oov_token="<UNK>")
            self.tokenizer.fit_on_texts([combined_text])
            self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences([combined_text])[0]
        sequences = np.array(sequences)
        
        # Create sliding window data
        x = []
        y = []
        for i in range(0, len(sequences) - self.seq_length):
            x.append(sequences[i:i + self.seq_length])
            y.append(sequences[i + self.seq_length])
        
        return np.array(x), np.array(y)
    
    def get_data(self):
        """Return the prepared dataset"""
        return (self.x_train, self.y_train, self.x_val, self.y_val, 
                self.x_test, self.y_test, self.input_shape, self.output_dim)
    
    def get_batch_iterator(self, split="train"):
        """Get batch iterator for the specified split"""
        if split == "train":
            x, y = self.x_train, self.y_train
        elif split == "val":
            x, y = self.x_val, self.y_val
        else:
            x, y = self.x_test, self.y_test
        
        # Create batches
        num_batches = len(x) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            yield x[start_idx:end_idx], y[start_idx:end_idx]

def get_wt2_dataset(seq_length=35, batch_size=20):
    """Helper function to get the WikiText-2 dataset"""
    dataset = WT2Dataset(seq_length=seq_length, batch_size=batch_size)
    return dataset.get_data()