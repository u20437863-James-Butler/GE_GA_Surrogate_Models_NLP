import tensorflow as tf
import numpy as np
import time

class MinTrainSurrogate:
    """
    Surrogate model that trains an RNN architecture for a limited number of epochs.
    Used as a faster fitness approximation in evolutionary algorithms for Neural Architecture Search.
    """
    def __init__(self, dataset, num_epochs=5, batch_size=128, verbose=0):
        """
        Initialize the minimum training surrogate model.
        
        Args:
            dataset: Dataset object containing training, validation and test data
            num_epochs: Fixed number of epochs to train each model
            batch_size: Batch size for training
            verbose: Verbosity level for training (0: silent, 1: progress bar, 2: one line per epoch)
        """
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Load dataset
        self.train_data = dataset.get_train_data()
        self.valid_data = dataset.get_valid_data()
        self.test_data = dataset.get_test_data()
        
        # Get input shape and output dimension from dataset
        self.input_shape = dataset.get_input_shape()
        self.output_dim = dataset.get_output_dim()
        
        # Metrics to track
        self.best_perplexity = float('inf')
        self.best_individual = None
    
    def calculate_perplexity(self, model, data):
        """Calculate perplexity from model predictions"""
        x, y = data
        loss = model.evaluate(x, y, verbose=0)[0]
        return np.exp(loss)
    
    def evaluate(self, individual):
        """
        Train the individual's model for a fixed number of epochs and return fitness.
        
        Args:
            individual: An Individual or CellBasedIndividual with a build_model method
            
        Returns:
            float: Fitness score (negative perplexity, higher is better)
        """
        start_time = time.time()
        
        # Build model from individual
        model = individual.build_model(self.input_shape, self.output_dim)
        
        # Train model for fixed number of epochs
        history = model.fit(
            self.train_data[0], self.train_data[1],
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=self.valid_data,
            verbose=self.verbose
        )
        
        # Calculate validation perplexity (lower is better)
        val_perplexity = self.calculate_perplexity(model, self.valid_data)
        
        # Track best model
        if val_perplexity < self.best_perplexity:
            self.best_perplexity = val_perplexity
            self.best_individual = individual.copy()
            
        # Training time
        training_time = time.time() - start_time
        
        # Set fitness score (negative perplexity, so higher is better)
        fitness = -val_perplexity
        individual.fitness = fitness
        
        # Print results if verbose
        if self.verbose > 0:
            print(f"Individual trained in {training_time:.2f}s | Val Perplexity: {val_perplexity:.2f} | Epochs: {self.num_epochs}")
            print(f"Architecture: {individual.__dict__}")
        
        return fitness
    
    def evaluate_population(self, population):
        """
        Evaluate a population of individuals.
        
        Args:
            population: List of Individual or CellBasedIndividual objects
            
        Returns:
            list: List of fitness scores
        """
        return [self.evaluate(individual) for individual in population]
    
    def get_best_individual(self):
        """Return the best individual found so far"""
        return self.best_individual