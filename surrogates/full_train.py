import tensorflow as tf
import numpy as np
import time
from datetime import datetime


class FullTrainer:
    """
    FullTrainer class to perform complete training on the best architecture
    found by neural architecture search.
    
    This class trains the best architecture for more epochs to get its true performance.
    """
    
    def __init__(self, dataset, num_epochs=40, batch_size=128, verbose=1):
        """
        Initialize the full trainer model.
        
        Args:
            dataset: Dataset object containing training, validation and test data
            num_epochs: Number of epochs to train the model (default: 40)
            batch_size: Batch size for training
            verbose: Verbosity level (0: silent, 1: show progress)
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Unpack the dataset tuple
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, self.input_shape, self.output_dim = dataset
        
        # Prepare data tuples
        self.train_data = (self.x_train, self.y_train)
        self.valid_data = (self.x_val, self.y_val)
        self.test_data = (self.x_test, self.y_test)
    
    def evaluate(self, individual, seed=None):
        """
        Fully train the individual's model and evaluate on test set.
        
        Args:
            individual: An Individual with a build_model method
            seed: Optional seed for weight initialization
            
        Returns:
            dict: Results including perplexity, training time, and model
        """
        # Get individual ID
        individual_id = individual.getId() if hasattr(individual, 'getId') else str(id(individual))
        
        # Get individual architecture (shorter summary)
        architecture = str(individual) if hasattr(individual, '__str__') else None
        arch_summary = architecture[:100] + '...' if architecture and len(architecture) > 100 else architecture
        
        print(f"\n{'='*50}")
        print(f"FULL TRAINING OF BEST ARCHITECTURE")
        print(f"Individual ID: {individual_id}")
        print(f"Architecture: {arch_summary}")
        print(f"Training for {self.num_epochs} epochs")
        print(f"{'='*50}\n")
        
        # Set seed if provided
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        # Create callbacks
        callbacks = []
        
        # Monitor training progress with simplified callback
        if self.verbose > 0:
            terminal_callback = self._create_terminal_callback()
            callbacks.append(terminal_callback)
            
            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Start timing
        start_time = time.time()
        
        # Build model from individual
        model = individual.build_model(self.input_shape, self.output_dim)
        
        # Train model
        history = model.fit(
            self.train_data[0], self.train_data[1],
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=self.valid_data,
            verbose=self.verbose,
            callbacks=callbacks
        )
        
        # Only evaluate on test data (skip re-evaluating train and validation)
        test_loss = model.evaluate(self.test_data[0], self.test_data[1], verbose=0)[0]
        test_perplexity = np.exp(test_loss)
        
        # Total training time
        training_time = time.time() - start_time
        
        # Print results (simplified)
        print(f"\n{'='*50}")
        print(f"FULL TRAINING RESULTS")
        print(f"{'='*50}")
        print(f"Training completed in {training_time:.2f}s")
        print(f"Test loss: {test_loss:.4f} (perplexity: {test_perplexity:.4f})")
        print(f"{'='*50}\n")
        
        # Package results (simplified)
        results = {
            'model': model,
            'history': history.history,
            'training_time': training_time,
            'test_loss': test_loss,
            'test_perplexity': test_perplexity,
            'individual': individual
        }
        
        return results

    def _create_terminal_callback(self):
        """Create a simplified terminal callback for training visualization"""
        class FullTrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.start_time = None
                self.best_val_loss = float('inf')
                self.num_epochs = None
            
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                self.num_epochs = self.params['epochs']
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                time_elapsed = time.time() - self.start_time
                
                # Basic metrics
                train_loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                
                # Only calculate perplexity for display
                train_ppl = np.exp(train_loss)
                val_ppl = np.exp(val_loss)
                
                # Simple best marker
                is_best = ""
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = "✓"
                
                # Format the output
                print(f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"loss={train_loss:.4f} (ppl={train_ppl:.2f}), "
                    f"val_loss={val_loss:.4f} (ppl={val_ppl:.2f}) "
                    f"{is_best} [{time_elapsed:.1f}s]")
        
        return FullTrainingCallback()