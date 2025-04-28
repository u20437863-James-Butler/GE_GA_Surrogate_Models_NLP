import numpy as np
import tensorflow as tf
import time
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
try:
    from surrogates.full_train import FullTrainer
except ImportError:
    sys.path.append(os.path.join(parent_dir, 'surrogates'))
    from full_train import FullTrainer

class Evaluator:
    """
    Evaluator class to run neural architecture search experiments.
    
    This class takes a surrogate model and dataset, then runs an 
    evolutionary optimizer to find optimal neural architectures.
    """
    
    def __init__(self, optimizer, dataset, max_runs=10, log_interval=1, full_runs=40, logger=None):
        """
        Initialize the evaluator.
        
        Args:
            surrogate: A surrogate model that evaluates individuals
            optimizer: An evolutionary algorithm optimizer
            dataset: Dataset to use for training/evaluation
            max_runs: Maximum number of runs to perform
            log_interval: How often to log progress
            full_runs: How many epochs to train the run results for
            logger: Logger object to record results
        """
        self.optimizer = optimizer
        self.dataset = dataset  # Store dataset for full training
        self.max_runs = max_runs
        self.log_interval = log_interval
        self.full_runs = full_runs
        self.logger = logger
        
        # Metrics tracking
        self.best_fitness = float('-inf')  # Higher is better
        self.best_individual = None
        self.fitness_history = []
        self.best_fitness_history = []
        self.run_count = 0
        
        # Track all full training results
        self.full_training_results = []
        
    def run(self, seed):
        """Run the neural architecture search experiment"""
        print(f"Starting neural architecture search with {self.optimizer.__class__.__name__}")
        print(f"Max runs: {self.max_runs}")
        
        total_start_time = time.time()
        
        for self.run_count in range(self.max_runs):
            start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Run {self.run_count + 1}/{self.max_runs}")
            print(f"{'='*50}")
            print("Generating initial population...")
            self.optimizer.population = self.optimizer.generate_population(seed=seed + self.run_count)
            
            # Get the best individual from this run
            individual = self.optimizer.evolve()
            
            # Update overall best individual if needed
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual
                print(f"New overall best fitness: {self.best_fitness:.4f}")
            
            # Update history
            self.fitness_history.append(individual.fitness)
            self.best_fitness_history.append(self.best_fitness)
            
            run_elapsed = time.time() - start_time
            print(f"Run {self.run_count + 1} of search completed in {run_elapsed:.1f}s")
            
            # Run full training on this run's best individual
            print(f"\nRunning full training for the best individual from run {self.run_count + 1}")
            print(f"Individual fitness from evolutionary search: {individual.fitness:.4f}")
            result = self.run_full_training(individual)
            
            # Store results with timing information
            run_result = {
                'run': self.run_count + 1,
                'individual': individual,
                'fitness': individual.fitness,
                'full_training_result': result,
                'time': run_elapsed
            }
            self.full_training_results.append(run_result)
            
            # Log results to CSV if logger is provided
            if self.logger:
                self.logger.log_run(self.run_count + 1, run_result)
            
            # Log progress at specified intervals
            if (self.run_count + 1) % self.log_interval == 0:
                print(f"\nProgress: {self.run_count + 1}/{self.max_runs} runs completed")
                print(f"Current overall best fitness: {self.best_fitness:.4f}")
        
        total_elapsed = time.time() - total_start_time
        
        print("\n" + "="*70)
        print("Neural Architecture Search Complete")
        print("="*70)
        print(f"Total runs: {self.run_count + 1}")
        print(f"Best fitness from evolutionary search: {self.best_fitness:.4f}")
        print(f"Total time: {total_elapsed:.1f}s")
        
        # Print summary of all runs
        print("\nSummary of all runs:")
        print(f"{'Run':<5} {'Evo Fitness':<15} {'Full Training Val Perplexity'}")
        print("-" * 50)
        for result in self.full_training_results:
            print(f"{result['run']:<5} {result['fitness']:<15.4f} {result['full_training_result']['val_perplexity']}")
        
        print(f"\nBest architecture: {self.best_individual}")
        
        # Log final results if logger is provided
        if self.logger:
            self.logger.log_final_results(self.best_individual, self.full_training_results, total_elapsed)
            
        return self.best_individual, self.full_training_results
    
    def run_full_training(self, individual):
        """
        Run full training on an individual.
        
        Args:
            individual: The individual to train
        
        Returns:
            dict: Results from full training
        """
        print(f"\nStarting full training of architecture...")        
        full_trainer = FullTrainer(self.dataset, num_epochs=self.full_runs)
        return full_trainer.evaluate(individual)