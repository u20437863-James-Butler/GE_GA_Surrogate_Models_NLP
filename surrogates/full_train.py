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
        
        # Get individual architecture
        architecture = str(individual) if hasattr(individual, '__str__') else None
        
        print(f"\n{'='*80}")
        print(f"FULL TRAINING OF BEST ARCHITECTURE")
        print(f"Individual ID: {individual_id}")
        print(f"Architecture: {architecture[:200] + '...' if architecture and len(architecture) > 200 else architecture}")
        print(f"Training for {self.num_epochs} epochs")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Set seed if provided
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        # Create callback for training monitoring
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        
        callbacks = []
        
        # Monitor training progress
        if self.verbose > 0:
            from tensorflow.keras.callbacks import CSVLogger, TensorBoard
            
            # Terminal callback for visual feedback
            terminal_callback = self._create_terminal_callback(individual_id, seed, architecture)
            callbacks.append(terminal_callback)
            
            # Early stopping to prevent overfitting
            early_stopping = EarlyStopping(
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
        
        # Calculate final metrics
        train_loss = model.evaluate(self.train_data[0], self.train_data[1], verbose=0)[0]
        val_loss = model.evaluate(self.valid_data[0], self.valid_data[1], verbose=0)[0]
        test_loss = model.evaluate(self.test_data[0], self.test_data[1], verbose=0)[0]
        
        # Calculate perplexities
        train_perplexity = np.exp(train_loss)
        val_perplexity = np.exp(val_loss)
        test_perplexity = np.exp(test_loss)
        
        # Total training time
        training_time = time.time() - start_time
        
        # Print results
        print(f"\n{'='*80}")
        print(f"FULL TRAINING RESULTS")
        print(f"{'='*80}")
        print(f"Training completed in {training_time:.2f}s")
        print(f"Training loss: {train_loss:.4f} (perplexity: {train_perplexity:.4f})")
        print(f"Validation loss: {val_loss:.4f} (perplexity: {val_perplexity:.4f})")
        print(f"Test loss: {test_loss:.4f} (perplexity: {test_perplexity:.4f})")
        print(f"{'='*80}\n")
        
        # Package results
        results = {
            'model': model,
            'history': history.history,
            'training_time': training_time,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity,
            'test_perplexity': test_perplexity,
            'individual': individual
        }
        
        return results
    
    def _create_terminal_callback(self, individual_id, seed, architecture):
        """Create a terminal callback for training visualization"""
        class FullTrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, individual_id, seed, architecture):
                super().__init__()
                self.individual_id = individual_id
                self.seed = seed
                self.architecture = architecture
                self.start_time = None
                self.best_perplexity = float('inf')
                self.best_epoch = 0
                self.num_epochs = None  # Will be set in on_train_begin
            
            def on_train_begin(self, logs=None):
                self.start_time = time.time()
                # Get total epochs from the params dictionary
                self.num_epochs = self.params['epochs']
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get('val_loss', None)
                time_elapsed = time.time() - self.start_time
                
                # Calculate metrics
                train_loss = logs.get('loss', None)
                train_ppl = np.exp(train_loss) if train_loss is not None else None
                val_ppl = np.exp(val_loss) if val_loss is not None else None
                
                # Track best performance
                is_best = ""
                if val_ppl is not None and val_ppl < self.best_perplexity:
                    self.best_perplexity = val_ppl
                    self.best_epoch = epoch + 1
                    is_best = "✓"
                
                # Format the output
                print(f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"loss={train_loss:.4f} ({train_ppl:.2f}), "
                    f"val_loss={val_loss:.4f} ({val_ppl:.2f}) "
                    f"{is_best} [{time_elapsed:.1f}s]")
        
        return FullTrainingCallback(individual_id, seed, architecture)