import tensorflow as tf
import numpy as np
import time
import os
import sys
from datetime import datetime

# Add parent directory to path to make imports work properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logs.minTrainLogger import MinTrainLogger

class MinTrainSurrogate:
    """
    Surrogate model that trains an RNN architecture for a limited number of epochs.
    Used as a faster fitness approximation in evolutionary algorithms for Neural Architecture Search.
    """
    def __init__(self, dataset, num_epochs=5, batch_size=128, verbose=0, dataset_name="default", timestamp=None):
        """
        Initialize the minimum training surrogate model.
        
        Args:
            dataset: Dataset object containing training, validation and test data
            num_epochs: Fixed number of epochs to train each model
            batch_size: Batch size for training
            verbose: Verbosity level for training (0: silent, 1: progress bar, 2: one line per epoch)
            dataset_name: Name of the dataset being used
            timestamp: Optional timestamp to use for log directory naming
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.dataset_name = dataset_name
        
        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        
        # Unpack the dataset tuple
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test, self.input_shape, self.output_dim = dataset
        
        # Prepare data tuples for easy access
        self.train_data = (self.x_train, self.y_train)
        self.valid_data = (self.x_val, self.y_val)
        self.test_data = (self.x_test, self.y_test)
        
        # Metrics to track
        self.best_perplexity = float('inf')
        self.best_individual = None
        
        # Initialize logger with dataset name and timestamp
        self.logger = MinTrainLogger(dataset_name=f"{dataset_name}_{timestamp}")
    
    def calculate_perplexity(self, model, data):
        """Calculate perplexity from model predictions"""
        x, y = data
        loss = model.evaluate(x, y, verbose=0)[0]
        return np.exp(loss)
    
    def evaluate(self, individual, log_filename=None):
        """
        Train the individual's model for a fixed number of epochs and return fitness.
        
        Args:
            individual: An Individual or CellBasedIndividual with a build_model method
            
        Returns:
            float: Fitness score (negative perplexity, higher is better)
        """
        start_time = time.time()

        # Get individual ID
        individual_id = individual.getId() if hasattr(individual, 'getId') else str(id(individual))

        # Create callback
        logger_callback = self.logger.create_epoch_callback(individual_id, log_filename)
        
        # Build model from individual
        model = individual.build_model(self.input_shape, self.output_dim)
        
        # Train model for fixed number of epochs
        history = model.fit(
            self.train_data[0], self.train_data[1],
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=self.valid_data,
            verbose=self.verbose,
            callbacks=[logger_callback]
        )
        
        # Calculate validation perplexity (lower is better)
        val_perplexity = self.calculate_perplexity(model, self.valid_data)
        
        # Track best model
        if val_perplexity < self.best_perplexity:
            self.best_perplexity = val_perplexity
            self.best_individual = individual.copy()
            
        # Training time
        training_time = time.time() - start_time

        # Log final results
        architecture_summary = str(individual)
        self.logger.log_final_result(
            individual_id, 
            val_perplexity, 
            training_time, 
            architecture_summary, 
            log_filename
        )
        
        # Set fitness score (negative perplexity, so higher is better)
        fitness = -val_perplexity
        individual.fitness = fitness
        
        # Print results if verbose
        if self.verbose > 0:
            print(f"Individual trained in {training_time:.2f}s | Val Perplexity: {val_perplexity:.2f} | Epochs: {self.num_epochs}")
            print(f"Architecture: {individual.__dict__}")
        
        return fitness
    
    def evaluate_population(self, population, base_log_filename=None):
        """
        Evaluate a population of individuals.
        
        Args:
            population: List of Individual or CellBasedIndividual objects
            
        Returns:
            list: List of fitness scores
        """
        fitness_scores = []
        
        for i, individual in enumerate(population):
            # Create individual log filename if base provided
            if base_log_filename:
                log_filename = f"{base_log_filename}_individual_{i}.csv"
            else:
                log_filename = None
                
            fitness = self.evaluate(individual, log_filename)
            fitness_scores.append(fitness)
            
        return fitness_scores
    
    def get_best_individual(self):
        """Return the best individual found so far"""
        return self.best_individual
    
    def get_best_perplexity(self):
        "Return the best perplexity found so far"
        return self.best_perplexity