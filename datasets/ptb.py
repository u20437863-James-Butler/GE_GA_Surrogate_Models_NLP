import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import requests
import io
import tarfile
from pathlib import Path

class PTBDataset:
    def __init__(self, seq_length=35, batch_size=20):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_dir = Path("datasets/ptb_data")
        self.tokenizer = None
        self.vocab_size = 0
        
        # Create datasets directory if it doesn't exist
        Path("datasets").mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Load and prepare data
        self.download_and_extract()
        self.x_train, self.y_train = self.load_and_prepare_data("train")
        self.x_val, self.y_val = self.load_and_prepare_data("valid")
        self.x_test, self.y_test = self.load_and_prepare_data("test")
        
        # Set input and output shapes for model building
        self.input_shape = (self.seq_length,)
        self.output_dim = self.vocab_size

    def download_and_extract(self):
        """Download and extract PTB dataset if not already available"""
        # Check if data files already exist
        required_files = [self.data_dir / f"ptb.{split}.txt" for split in ["train", "valid", "test"]]
        
        if all(file.exists() for file in required_files):
            print("PTB dataset files already exist. Skipping download.")
            return
            
        print("Downloading Penn TreeBank dataset...")
        url = "https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
        response = requests.get(url)
        
        # Extract files
        tar = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
        tar.extractall()
        tar.close()
        
        # Move relevant files
        source_dir = Path("simple-examples/data")
        for split in ["train", "valid", "test"]:
            source_file = source_dir / f"ptb.{split}.txt"
            target_file = self.data_dir / f"ptb.{split}.txt"
            if source_file.exists():
                with open(source_file, 'r', encoding='utf-8') as src, \
                     open(target_file, 'w', encoding='utf-8') as tgt:
                    tgt.write(src.read())
        
        # Clean up extracted files
        import shutil
        if Path("simple-examples").exists():
            shutil.rmtree("simple-examples")
            
        print("Download and extraction complete.")
    
    def load_and_prepare_data(self, split):
        """Load and prepare data for the specified split"""
        file_path = self.data_dir / f"ptb.{split}.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize on first pass only
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(oov_token="<UNK>")
            self.tokenizer.fit_on_texts([text])
            self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Convert to sequences and create input-target pairs
        sequences = self.tokenizer.texts_to_sequences([text])[0]
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

def get_ptb_dataset(seq_length=35, batch_size=20):
    """Helper function to get the PTB dataset"""
    dataset = PTBDataset(seq_length=seq_length, batch_size=batch_size)
    return dataset.get_data()