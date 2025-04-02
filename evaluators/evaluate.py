import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

class Evaluator:
    """
    Evaluator class to run neural architecture search experiments.
    
    This class takes a surrogate model and dataset, then runs an 
    evolutionary optimizer to find optimal neural architectures.
    """
    
    def __init__(self, surrogate, optimizer, max_evaluations=100, log_interval=10):
        """
        Initialize the evaluator.
        
        Args:
            surrogate: A surrogate model that evaluates individuals
            optimizer: An evolutionary algorithm optimizer
            max_evaluations: Maximum number of evaluations to perform
            log_interval: How often to log progress
        """
        self.surrogate = surrogate
        self.optimizer = optimizer
        self.max_evaluations = max_evaluations
        self.log_interval = log_interval
        
        # Get dataset input shape and output dimensions from surrogate
        self.input_shape = surrogate.input_shape
        self.output_dim = surrogate.output_dim
        
        # Metrics tracking
        self.best_fitness = float('-inf')  # Higher is better
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluation_count = 0
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
        self.evaluation_count += 1
        
        # Track best individual
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_individual = individual.copy()
            
        self.fitness_history.append(fitness)
        self.best_fitness_history.append(self.best_fitness)
        
        # Log progress
        if self.evaluation_count % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            perplexity = -self.best_fitness  # Convert fitness back to perplexity
            print(f"Eval: {self.evaluation_count}/{self.max_evaluations} | "
                  f"Best Perplexity: {perplexity:.2f} | "
                  f"Time: {elapsed:.1f}s")
            
        return fitness
        
    def run(self):
        """Run the neural architecture search experiment"""
        print(f"Starting neural architecture search with {self.optimizer.__class__.__name__}")
        print(f"Surrogate model: {self.surrogate.__class__.__name__}")
        print(f"Max evaluations: {self.max_evaluations}")
        
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
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Best validation perplexity: {perplexity:.2f}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Best architecture: {self.best_individual.__dict__}")
        
        # Plot progress
        self.plot_progress()
        
        return self.best_individual
    
    def evaluate_population(self, population):
        """Evaluate an entire population using the surrogate model"""
        fitnesses = []
        for individual in population:
            # Respect evaluation limit
            if self.evaluation_count >= self.max_evaluations:
                break
                
            fitness = self.evaluate_individual(individual)
            fitnesses.append(fitness)
            
        return fitnesses
    
    def plot_progress(self):
        """Plot optimization progress"""
        plt.figure(figsize=(10, 6))
        
        # Convert fitness back to perplexity for plotting
        perplexity_history = [-fitness for fitness in self.fitness_history]
        best_perplexity_history = [-fitness for fitness in self.best_fitness_history]
        
        plt.plot(perplexity_history, 'o', alpha=0.3, label='Individual Perplexity')
        plt.plot(best_perplexity_history, '-', label='Best Perplexity')
        
        plt.xlabel('Evaluation')
        plt.ylabel('Validation Perplexity (lower is better)')
        plt.title('Neural Architecture Search Progress')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/nas_progress.png')
        plt.close()

def run_experiment(dataset, surrogate, optimizer, max_evaluations=100):
    """
    Run a neural architecture search experiment.
    
    Args:
        dataset: Dataset to use for training/evaluation
        surrogate: Surrogate model for fitness evaluation
        optimizer: Evolutionary algorithm to use
        max_evaluations: Maximum number of evaluations to perform
        
    Returns:
        best_individual: The best architecture found
    """
    evaluator = Evaluator(surrogate, optimizer, max_evaluations)
    return evaluator.run()