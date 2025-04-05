import os
import csv
import time
from datetime import datetime
import tensorflow as tf
import numpy as np

class MinTrainLogger:
    """
    Logger for tracking perplexity and other metrics during model training in MinTrainSurrogate.
    Creates CSV logs in the mintrain_logs_{dataset_name}_{timestamp} directory.
    """
    def __init__(self, dataset_name="default", log_dir=None, timestamp=None):
        """
        Initialize the MinTrainLogger.
        
        Args:
            dataset_name: Name of the dataset being used, included in log directory name
            log_dir: Directory to save logs. Defaults to './logs/mintrain_logs_{dataset_name}_{timestamp}/'
            timestamp: Optional timestamp to use for the log directory. If None, generates a new one.
        """
        # Use provided timestamp or generate a new one
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        
        # Set up log directory with dataset name and timestamp
        if log_dir is None:
            # Create path relative to this file's location
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_dir = os.path.join(parent_dir, f'mintrain_logs_{dataset_name}_{timestamp}')
        else:
            # If custom dir provided, still append dataset name and timestamp
            self.log_dir = os.path.join(log_dir, f'mintrain_logs_{dataset_name}_{timestamp}')
            
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a summary file for all runs
        self.summary_file = os.path.join(self.log_dir, 'training_summary.csv')
        with open(self.summary_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'individual_id', 
                'seed', 
                'timestamp', 
                'final_train_loss', 
                'final_val_loss', 
                'final_perplexity', 
                'best_epoch', 
                'best_perplexity', 
                'training_time'
            ])
    
    def create_epoch_callback(self, individual_id, seed=None, log_filename=None, architecture=None):
        """
        Create a Keras callback to log metrics after each epoch.
        
        Args:
            individual_id: ID of the individual being trained.
            seed: Optional seed used for weight initialization.
            log_filename: Optional CSV filename. If None, uses timestamp.
            architecture: A string representing the model architecture, typically from individual.getIdLong().
            
        Returns:
            tf.keras.callbacks.Callback: Callback for logging metrics.
        """
        # Create filename if not provided
        if log_filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"model_{individual_id}_seed_{seed}_{current_time}.csv"
        
        # Ensure filename has .csv extension
        if not log_filename.endswith('.csv'):
            log_filename += '.csv'
            
        # Full path to log file
        log_path = os.path.join(self.log_dir, log_filename)
        
        # Create CSV file with headers (added 'architecture' column)
        with open(log_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'individual_id', 
                'seed', 
                'epoch', 
                'train_loss', 
                'val_loss', 
                'perplexity', 
                'time_elapsed',
                'architecture'
            ])
        
        # Create custom callback for logging
        class PerplexityLogger(tf.keras.callbacks.Callback):
            def __init__(self, log_path, individual_id, seed, summary_file, architecture):
                super().__init__()
                self.log_path = log_path
                self.individual_id = individual_id
                self.seed = seed
                self.start_time = None
                self.summary_file = summary_file
                self.best_perplexity = float('inf')
                self.best_epoch = 0
                self.architecture = architecture
                    
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                    
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                train_loss = logs.get('loss', None)
                val_loss = logs.get('val_loss', None)
                time_elapsed = time.time() - self.start_time
                    
                # Calculate perplexity from validation loss
                if val_loss is not None:
                    perplexity = np.exp(val_loss)
                else:
                    perplexity = None
                        
                # Track best perplexity
                if perplexity is not None and perplexity < self.best_perplexity:
                    self.best_perplexity = perplexity
                    self.best_epoch = epoch + 1
                    
                # Append to CSV including the architecture column
                with open(self.log_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.individual_id,
                        self.seed,
                        epoch + 1,  # 1-based epoch indexing
                        train_loss,
                        val_loss,
                        perplexity,
                        time_elapsed,
                        self.architecture
                    ])
                        
            def on_train_end(self, logs=None):
                logs = logs or {}
                training_time = time.time() - self.start_time
                    
                # Get final metrics
                final_train_loss = logs.get('loss', None)
                final_val_loss = logs.get('val_loss', None)
                    
                # Calculate final perplexity
                if final_val_loss is not None:
                    final_perplexity = np.exp(final_val_loss)
                else:
                    final_perplexity = None
                    
                # Add to summary file
                with open(self.summary_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.individual_id,
                        self.seed,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        final_train_loss,
                        final_val_loss,
                        final_perplexity,
                        self.best_epoch,
                        self.best_perplexity,
                        training_time
                    ])
                    
        return PerplexityLogger(log_path, individual_id, seed, self.summary_file, architecture)

    
    def log_final_result(self, individual_id, perplexity, training_time, architecture_summary, seed=None, log_filename=None):
        """
        Log the final evaluation results for an individual.
        
        Args:
            individual_id: ID of the individual
            perplexity: Final validation perplexity
            training_time: Total training time in seconds
            architecture_summary: Summary of the model architecture
            seed: Optional seed used for weight initialization
            log_filename: Optional CSV filename. If None, uses timestamp
        """
        # Create filename if not provided
        if log_filename is None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"final_model_{individual_id}_seed_{seed}_{current_time}.csv"
        
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
                writer.writerow([
                    'individual_id', 
                    'seed', 
                    'perplexity', 
                    'training_time', 
                    'timestamp', 
                    'architecture'
                ])
            
            # Write data row
            writer.writerow([
                individual_id,
                seed,
                perplexity,
                training_time,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                architecture_summary
            ])
            
    def log_run_summary(self, run_id, seed, best_individual, best_perplexity, elapsed_time):
        """
        Log a summary of a complete run with multiple individuals.
        
        Args:
            run_id: ID or name of the run
            seed: Seed used for weight initialization
            best_individual: Best individual found in this run
            best_perplexity: Best perplexity achieved
            elapsed_time: Total run time in seconds
        """
        # Path to run summary file
        summary_path = os.path.join(self.log_dir, 'run_summaries.csv')
        
        # Check if file exists to write headers
        file_exists = os.path.isfile(summary_path)
        
        # Write to CSV
        with open(summary_path, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers if file is new
            if not file_exists:
                writer.writerow([
                    'run_id', 
                    'seed', 
                    'best_individual_id', 
                    'best_perplexity', 
                    'elapsed_time', 
                    'timestamp'
                ])
            
            # Write data row
            writer.writerow([
                run_id,
                seed,
                best_individual.getId() if best_individual else 'None',
                best_perplexity,
                elapsed_time,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])