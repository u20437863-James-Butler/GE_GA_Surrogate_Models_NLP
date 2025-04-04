import os
import sys
import numpy as np
import time

# Add the parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the dataset functions from the datasets directory
try:
    from datasets.ptb import get_ptb_dataset
    from datasets.wt2 import get_wt2_dataset
except ImportError:
    print("Could not import directly. Trying alternative import method...")
    # Alternative import if datasets is not a proper package
    sys.path.append(os.path.join(parent_dir, 'datasets'))
    from ptb import get_ptb_dataset
    from wt2 import get_wt2_dataset

def test_dataset(name, dataset_fn, seq_length=35, batch_size=20):
    """Test if a dataset can be loaded and print basic information about it"""
    print(f"\nTesting {name} dataset...")
    start_time = time.time()
    
    try:
        # Load the dataset
        x_train, y_train, x_val, y_val, x_test, y_test, input_shape, output_dim = dataset_fn(
            seq_length=seq_length, batch_size=batch_size
        )
        
        # Print dataset information
        print(f"  Dataset loaded successfully in {time.time() - start_time:.2f} seconds")
        print(f"  Vocabulary size: {output_dim}")
        print(f"  Input shape: {input_shape}")
        print(f"  Training samples: {len(x_train)}")
        print(f"  Validation samples: {len(x_val)}")
        print(f"  Test samples: {len(x_test)}")
        
        # Verify shapes
        print(f"  Training data shape: {x_train.shape}")
        print(f"  Training labels shape: {y_train.shape}")
        
        # Display sample data
        print(f"  Sample input (first sequence):")
        print(f"    {x_train[0][:10]}... (truncated)")
        print(f"  Sample target: {y_train[0]}")
        
        return True
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return False

def main():
    """Main function to test all datasets"""
    print("Dataset Testing Script")
    print("=====================")
    
    # Test PTB dataset
    ptb_success = test_dataset("Penn TreeBank (PTB)", get_ptb_dataset)
    
    # Test WikiText-2 dataset
    wt2_success = test_dataset("WikiText-2", get_wt2_dataset)
    
    # Summary
    print("\nSummary:")
    print(f"  PTB dataset: {'✓ Loaded successfully' if ptb_success else '✗ Failed to load'}")
    print(f"  WikiText-2 dataset: {'✓ Loaded successfully' if wt2_success else '✗ Failed to load'}")

if __name__ == "__main__":
    main()