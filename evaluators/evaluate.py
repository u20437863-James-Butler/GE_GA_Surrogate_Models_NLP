import numpy as np
import tensorflow as tf
import time
import os

class Evaluator:
    """
    Evaluator class to run neural architecture search experiments.
    
    This class takes a surrogate model and dataset, then runs an 
    evolutionary optimizer to find optimal neural architectures.
    """
    
    def __init__(self, surrogate, optimizer, max_runs=10, log_interval=11):
        """
        Initialize the evaluator.
        
        Args:
            surrogate: A surrogate model that evaluates individuals
            optimizer: An evolutionary algorithm optimizer
            max_runs: Maximum number of runs to perform
            log_interval: How often to log progress
        """
        self.surrogate = surrogate
        self.optimizer = optimizer
        self.max_runs = max_runs
        self.log_interval = log_interval
        
        # Get dataset input shape and output dimensions from surrogate
        self.input_shape = surrogate.input_shape
        self.output_dim = surrogate.output_dim
        
        # Metrics tracking
        self.best_fitness = float('-inf')  # Higher is better
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.run_count = 0
        self.start_time = None
        
    def configure_optimizer(self):
        """Configure the optimizer with dataset parameters"""
        if not hasattr(self.optimizer, 'input_shape'):
            self.optimizer.input_shape = self.input_shape
        if not hasattr(self.optimizer, 'output_dim'):
            self.optimizer.output_dim = self.output_dim
            
    def evaluate_individual(self, individual):
        """Evaluate a single individual using the surrogate model"""
        fitness = self.surrogate.evaluate(individual)
        self.run_count += 1
        
        # Track best individual
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = individual.copy()
            
        self.fitness_history.append(fitness)
        self.best_fitness_history.append(self.best_fitness)
        
        # Log progress
        if self.run_count % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            perplexity = -self.best_fitness  # Convert fitness back to perplexity
            print(f"Run: {self.run_count}/{self.max_runs} | "
                  f"Best Perplexity: {perplexity:.2f} | "
                  f"Time: {elapsed:.1f}s")
            
        return fitness
        
    def run(self):
        """Run the neural architecture search experiment"""
        print(f"Starting neural architecture search with {self.optimizer.__class__.__name__}")
        print(f"Surrogate model: {self.surrogate.__class__.__name__}")
        print(f"Max runs: {self.max_runs}")
        
        self.start_time = time.time()
        self.configure_optimizer()
        
        # Replace direct evaluation in optimizer with surrogate evaluation
        self.original_evaluate_population = self.optimizer.evaluate_population
        self.optimizer.evaluate_population = self.evaluate_population
        
        # Run the optimizer
        best = self.optimizer.evolve()
        
        # Restore original method
        self.optimizer.evaluate_population = self.original_evaluate_population
        
        # Report final results
        elapsed = time.time() - self.start_time
        perplexity = -self.best_fitness
        
        print("\nNeural Architecture Search Complete")
        print(f"Total runs: {self.run_count}")
        print(f"Best validation perplexity: {perplexity:.2f}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Best architecture: {self.best_individual.__dict__}")
        
        return self.best_individual
    
    def evaluate_population(self, population):
        """Evaluate an entire population using the surrogate model"""
        fitnesses = []
        for individual in population:
            # Respect run limit
            if self.run_count >= self.max_runs:
                break
                
            fitness = self.evaluate_individual(individual)
            fitnesses.append(fitness)
            
        return fitnesses


def run_experiment(dataset, surrogate, optimizer, max_runs=10):
    """
    Run a neural architecture search experiment.
    
    Args:
        dataset: Dataset to use for training/evaluation
        surrogate: Surrogate model for fitness evaluation
        optimizer: Evolutionary algorithm to use
        max_runs: Maximum number of runs to perform
        
    Returns:
        best_individual: The best architecture found
    """
    evaluator = Evaluator(surrogate, optimizer, max_runs)
    return evaluator.run()