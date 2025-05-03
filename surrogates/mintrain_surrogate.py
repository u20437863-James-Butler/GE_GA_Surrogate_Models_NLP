import tensorflow as tf
import numpy as np
import time
from datetime import datetime

class TerminalTrainingCallback(tf.keras.callbacks.Callback):
    """
    Callback for tracking and displaying training metrics in the terminal.
    Inspired by MinTrainLogger but without file logging.
    """
    def __init__(self, individual_id, seed=None, architecture=None):
        """
        Initialize the terminal training callback.
        
        Args:
            individual_id: ID of the individual being trained
            seed: Optional seed used for weight initialization
            architecture: A string representing the model architecture
        """
        super().__init__()
        self.individual_id = individual_id
        self.seed = seed
        self.architecture = architecture
        self.start_time = None
        self.best_perplexity = float('inf')
        self.best_epoch = 0
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"\n{'-'*80}")
        print(f"Training individual: {self.individual_id}")
        print(f"Seed: {self.seed if self.seed is not None else 'Not specified'}")
        print(f"Architecture: {self.architecture[:200] + '...' if self.architecture and len(self.architecture) > 200 else self.architecture}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # train_loss = logs.get('loss', None)
        val_loss = logs.get('val_loss', None)
        # time_elapsed = time.time() - self.start_time
        
        # Calculate perplexity from validation loss
        if val_loss is not None:
            perplexity = np.exp(val_loss)
            
            # Track best perplexity
            # is_best = ""
            if perplexity < self.best_perplexity:
                self.best_perplexity = perplexity
                self.best_epoch = epoch + 1
                # is_best = "✓"
        else:
            perplexity = None
            # is_best = ""
        
        # # Format values for display
        # train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        # val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        # perplexity_str = f"{perplexity:.4f}" if perplexity is not None else "N/A"
        
        # # Print progress
        # print(f"\n{'-'*80}")
        # print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Val Loss':^12} | {'Perplexity':^12} | {'Best':^8} | {'Time':^10}")
        # print(f"{'-'*80}")
        # print(f"{epoch+1:6d} | {train_loss_str:12} | {val_loss_str:12} | {perplexity_str:12} | {is_best:^8} | {time_elapsed:.2f}s")
    
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
        
        # Print summary
        print(f"{'-'*80}")
        print(f"Training completed in {training_time:.2f}s")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best perplexity: {self.best_perplexity:.4f}")
        print(f"Final train loss: {final_train_loss:.4f}" if final_train_loss is not None else "Final train loss: N/A")
        print(f"Final val loss: {final_val_loss:.4f}" if final_val_loss is not None else "Final val loss: N/A")
        print(f"Final perplexity: {final_perplexity:.4f}" if final_perplexity is not None else "Final perplexity: N/A")
        print(f"{'-'*80}\n")


class SimplifiedMinTrainSurrogate:
    """
    Simplified surrogate model that trains RNN architectures for a fixed number of epochs.
    Used as a faster fitness approximation in evolutionary algorithms for Neural Architecture Search.
    """
    def __init__(self, dataset, num_epochs=1, batch_size=128, verbose=1):
        """
        Initialize the simplified minimum training surrogate model.
        
        Args:
            dataset: Dataset object containing training, validation and test data
            num_epochs: Fixed number of epochs to train each model (default: 1)
                        Set to 0 to skip training and only evaluate model on test set
            batch_size: Batch size for training
            verbose: Verbosity level (0: silent, 1: show training progress)
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
        
        # Track best model
        self.best_perplexity = float('inf')
        self.best_individual = None
        
        # Initialize fitness cache dictionary
        self.fitness_cache = {}
        
        # Generate timestamp for reference
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*80}")
        print(f"Initialized SimplifiedMinTrainSurrogate")
        print(f"Timestamp: {self.timestamp}")
        if num_epochs == 0:
            print(f"Mode: Evaluation only (0 epochs)")
        else:
            print(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
        print(f"Input shape: {self.input_shape}, Output dimension: {self.output_dim}")
        print(f"Training samples: {len(self.x_train)}, Validation samples: {len(self.x_val)}, Test samples: {len(self.x_test)}")
        print(f"{'='*80}\n")
    
    def calculate_perplexity(self, model, data):
        """Calculate perplexity from model predictions"""
        x, y = data
        loss = model.evaluate(x, y, verbose=0)[0]
        return np.exp(loss)
    
    def evaluate(self, individual, seed=None):
        """
        Train the individual's model for a fixed number of epochs and return fitness.
        If num_epochs=0, skip training and only evaluate on test set.
        If the individual has been evaluated before, return the cached fitness value.
        
        Args:
            individual: An Individual with a build_model method
            seed: Optional seed for weight initialization
            
        Returns:
            float: Fitness score (negative perplexity, higher is better)
        """
        # Get individual ID
        individual_id = individual.getGenericId() if hasattr(individual, 'getId') else str(id(individual))
        
        # Check if individual is in cache
        if individual_id in self.fitness_cache:
            fitness = self.fitness_cache[individual_id]
            individual.fitness = fitness
            return fitness
        
        # Get individual architecture
        architecture = str(individual) if hasattr(individual, '__str__') else None
        
        # Set seed if provided
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
        # Build model from individual
        model = individual.build_model(self.input_shape, self.output_dim)
        
        # Skip training if num_epochs is 0
        if self.num_epochs == 0:
            if self.verbose > 0:
                print(f"\n{'-'*80}")
                print(f"Evaluating individual without training: {individual_id}")
                print(f"Seed: {seed if seed is not None else 'Not specified'}")
                print(f"Architecture: {architecture[:200] + '...' if architecture and len(architecture) > 200 else architecture}")
                
                # Calculate test set perplexity
                test_start_time = time.time()
                test_perplexity = self.calculate_perplexity(model, self.test_data)
                test_time = time.time() - test_start_time
                
                print(f"Test perplexity: {test_perplexity:.4f}")
                print(f"Evaluation completed in {test_time:.2f}s")
                print(f"{'-'*80}\n")
                
            else:
                # Just evaluate on test set without verbose output
                test_perplexity = self.calculate_perplexity(model, self.test_data)
                
            # For zero-epoch mode, use test perplexity as fitness
            val_perplexity = test_perplexity
            
            # Set placeholder values for metrics that would normally come from training
            individual.train_loss = 0.0
            individual.val_loss = 0.0
            individual.best_epoch = 0
            
        else:
            # Create callback if verbose
            callbacks = []
            if self.verbose > 0:
                terminal_callback = TerminalTrainingCallback(
                    individual_id=individual_id,
                    seed=seed,
                    architecture=architecture
                )
                callbacks.append(terminal_callback)
            
            # Train model
            history = model.fit(
                self.train_data[0], self.train_data[1],
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                validation_data=self.valid_data,
                verbose=self.verbose,
                callbacks=callbacks
            )
            
            # Calculate validation perplexity (lower is better)
            val_perplexity = self.calculate_perplexity(model, self.valid_data)
            
            # Store metrics on individual
            individual.train_loss = history.history['loss'][-1] if 'loss' in history.history else None
            individual.val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None
            individual.best_epoch = callbacks[0].best_epoch if callbacks else self.num_epochs
            
        # Track best model
        if val_perplexity < self.best_perplexity:
            self.best_perplexity = val_perplexity
            self.best_individual = individual.copy()
            print(f"New best model found! Perplexity: {val_perplexity:.4f}")
        
        # Store test perplexity on individual
        individual.test_perplexity = self.calculate_perplexity(model, self.test_data)
        
        # Set fitness score (negative perplexity, so higher is better)
        fitness = -val_perplexity
        individual.fitness = fitness
        
        # Store in cache
        self.fitness_cache[individual_id] = fitness
        
        return fitness
    
    def evaluate_population(self, population, seed=None):
        """
        Evaluate a population of individuals.
        
        Args:
            population: List of Individual objects
            seed: Optional seed for weight initialization
            
        Returns:
            list: List of fitness scores
        """
        fitness_scores = []
        population_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"Evaluating population of {len(population)} individuals")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.num_epochs == 0:
            print(f"Mode: Evaluation only (0 epochs)")
        else:
            print(f"Training for {self.num_epochs} epochs per individual")
        print(f"{'='*80}")
        
        for i, individual in enumerate(population):
            print(f"\nIndividual {i+1}/{len(population)}")
            
            # Evaluate with individual seed if available, otherwise use provided seed
            ind_seed = individual.seed if hasattr(individual, 'seed') else seed
            fitness = self.evaluate(individual, seed=ind_seed)
            fitness_scores.append(fitness)
        
        # Sort individuals by fitness
        population_time = time.time() - population_start_time
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort by fitness (descending)
        
        print(f"\n{'='*80}")
        print(f"Population evaluation completed in {population_time:.2f}s")
        print(f"{'='*80}")
        print("\nTop performers:")
        for i in range(min(5, len(population))):
            idx = sorted_indices[i]
            ind = population[idx]
            print(f"  #{i+1}: ID={ind.getId() if hasattr(ind, 'getId') else 'Unknown'} | "
                  f"Fitness={fitness_scores[idx]:.4f} | Perplexity={-fitness_scores[idx]:.4f}")
        
        print(f"\nBest perplexity overall: {self.best_perplexity:.4f}")
        print(f"{'='*80}\n")
        
        return fitness_scores
    
    def get_best_individual(self):
        """Return the best individual found so far"""
        return self.best_individual
    
    def get_best_perplexity(self):
        """Return the best perplexity found so far"""
        return self.best_perplexity