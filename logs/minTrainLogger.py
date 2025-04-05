import os
import csv
import time
from datetime import datetime
import tensorflow as tf
import numpy as np

class MinTrainLogger:
    """
    Logger for tracking perplexity and other metrics during model training in MinTrainSurrogate.
    Creates CSV logs in the mintrain_logs_{dataset_name} directory.
    """
    def __init__(self, dataset_name="default", log_dir=None):
        """
        Initialize the MinTrainLogger.
        
        Args:
            dataset_name: Name of the dataset being used, included in log directory name
            log_dir: Directory to save logs. Defaults to './logs/mintrain_logs_{dataset_name}/'
        """
        # Set up log directory with dataset name
        if log_dir is None:
            # Create path relative to this file's location
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_dir = os.path.join(parent_dir, f'mintrain_logs_{dataset_name}')
        else:
            # If custom dir provided, still append dataset name
            self.log_dir = os.path.join(log_dir, f'mintrain_logs_{dataset_name}')
            
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def create_epoch_callback(self, individual_id, log_filename=None):
        """
        Create a Keras callback to log metrics after each epoch.
        
        Args:
            individual_id: ID of the individual being trained
            log_filename: Optional CSV filename. If None, uses timestamp
            
        Returns:
            tf.keras.callbacks.Callback: Callback for logging metrics
        """
        # Create filename if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"model_{individual_id}_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not log_filename.endswith('.csv'):
            log_filename += '.csv'
            
        # Full path to log file
        log_path = os.path.join(self.log_dir, log_filename)
        
        # Create CSV file with headers
        with open(log_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['individual_id', 'epoch', 'train_loss', 'val_loss', 'perplexity'])
        
        # Create custom callback for logging
        class PerplexityLogger(tf.keras.callbacks.Callback):
            def __init__(self, log_path, individual_id):
                super().__init__()
                self.log_path = log_path
                self.individual_id = individual_id
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                train_loss = logs.get('loss', None)
                val_loss = logs.get('val_loss', None)
                
                # Calculate perplexity from validation loss
                if val_loss is not None:
                    perplexity = np.exp(val_loss)
                else:
                    perplexity = None
                
                # Append to CSV
                with open(self.log_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.individual_id,
                        epoch + 1,  # 1-based epoch indexing
                        train_loss,
                        val_loss,
                        perplexity
                    ])
                    
        return PerplexityLogger(log_path, individual_id)
    
    def log_final_result(self, individual_id, perplexity, training_time, architecture_summary, log_filename=None):
        """
        Log the final evaluation results for an individual.
        
        Args:
            individual_id: ID of the individual
            perplexity: Final validation perplexity
            training_time: Total training time in seconds
            architecture_summary: Summary of the model architecture
            log_filename: Optional CSV filename. If None, uses timestamp
        """
        # Create filename if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"final_model_{individual_id}_{timestamp}.csv"
        
        # Ensure filename has .csv extension
        if not log_filename.endswith('.csv'):
            log_filename += '.csv'
            
        # Full path to log file
        log_path = os.path.join(self.log_dir, 'final_results.csv')
        
        # Check if file exists to write headers
        file_exists = os.path.isfile(log_path)
        
        # Write to CSV
        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow(['individual_id', 'perplexity', 'training_time', 'timestamp', 'architecture'])
            
            # Write data row
            writer.writerow([
                individual_id,
                perplexity,
                training_time,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                architecture_summary
            ])