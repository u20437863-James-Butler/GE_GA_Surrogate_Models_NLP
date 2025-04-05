import numpy as np
import tensorflow as tf
import time
import os
from datetime import datetime

class SurrEvaluator:
    """
    Evaluator class to run neural architecture search experiments.
    
    This class takes a surrogate model and dataset, then evaluates a fixed
    population of neural architectures across multiple seeds.
    """
    
    def __init__(self, optimizer, num_runs=5, log_interval=1, seeds=None):
        """
        Initialize the evaluator.
        
        Args:
            optimizer: An evolutionary algorithm optimizer
            num_runs: Number of runs with different seeds to perform
            log_interval: How often to log progress
            seeds: List of seeds to use for runs (generated if None)
        """
        self.optimizer = optimizer
        self.surrogate = optimizer.surrogate
        self.log_interval = log_interval
        self.num_runs = num_runs
        
        # Set up seeds for multiple runs
        if seeds is None:
            self.seeds = [42 + i for i in range(num_runs)]
        else:
            self.seeds = seeds[:num_runs]  # Use provided seeds
            
        # Get dataset input shape and output dimensions from surrogate
        self.input_shape = self.surrogate.input_shape
        self.output_dim = self.surrogate.output_dim
        
        # Metrics tracking
        self.best_fitness = float('-inf')  # Higher is better
        self.best_individual = None
        
        # Results across all runs
        self.run_results = []
        
        # Generate population once to reuse across runs
        self.configure_optimizer()
        self.population = None  # Will be generated in the first run
        
    def configure_optimizer(self):
        """Configure the optimizer with dataset parameters"""
        if not hasattr(self.optimizer, 'input_shape'):
            self.optimizer.input_shape = self.input_shape
        if not hasattr(self.optimizer, 'output_dim'):
            self.optimizer.output_dim = self.output_dim
    
    def run_single_run(self, run_index, seed):
        """
        Run a single evaluation with a specific seed.
        
        Args:
            run_index: Index of the current run
            seed: Seed for weight initialization
            
        Returns:
            dict: Results of this run
        """
        print(f"\nRun {run_index+1}/{self.num_runs} with seed {seed}")
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{run_index+1}_seed_{seed}_{timestamp}"
        
        # Create a new surrogate with the timestamp for unique log directory
        surrogate_class = self.surrogate.__class__
        new_surrogate = surrogate_class(
            dataset=self.surrogate._dataset if hasattr(self.surrogate, '_dataset') else 
                   (self.surrogate.x_train, self.surrogate.y_train, 
                    self.surrogate.x_val, self.surrogate.y_val,
                    self.surrogate.x_test, self.surrogate.y_test,
                    self.surrogate.input_shape, self.surrogate.output_dim),
            num_epochs=self.surrogate.num_epochs,
            batch_size=self.surrogate.batch_size,
            verbose=self.surrogate.verbose,
            dataset_name=self.surrogate.dataset_name,
            timestamp=timestamp
        )
        
        # Point optimizer to new surrogate
        self.optimizer.surrogate = new_surrogate
        
        # Generate population only on first run, then reuse it
        if self.population is None:
            print("Generating initial population (will be reused across all runs)")
            self.population = self.optimizer.generate_population(seed=42)
        else:
            print("Reusing initial population from first run")
            
        # Set the seed for weight initialization
        start_time = time.time()
        
        # Deep copy the population to avoid modifying the original
        population_copy = []
        for individual in self.population:
            population_copy.append(individual.copy())
            
        # Evaluate population with seed affecting only weights
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        print(f"Training all individuals with seed {seed} for {self.surrogate.num_epochs} epochs")
        fitness_scores = self.optimizer.evaluate_only(
            population=population_copy,
            seed=seed, 
            base_log_filename=run_name
        )
        
        elapsed = time.time() - start_time
        

        # Get best individual from this run by finding the index with the highest fitness score
        best_index = np.argmax(fitness_scores) if fitness_scores else None
        if best_index is not None:
            best_individual = population_copy[best_index]
            best_fitness = fitness_scores[best_index]
        else:
            best_individual = None
            best_fitness = float('-inf')

        best_perplexity = -best_fitness

        
        print(f"Run {run_index+1} complete in {elapsed:.1f}s")
        print(f"Best Perplexity: {best_perplexity:.2f}")
        
        # Store run results
        run_result = {
            'seed': seed,
            'timestamp': timestamp,
            'best_individual': best_individual.copy() if best_individual else None,
            'best_fitness': best_fitness,
            'best_perplexity': best_perplexity,
            'elapsed_time': elapsed,
            'fitness_scores': fitness_scores.copy() if fitness_scores else []
        }
        
        return run_result
        
    def run(self):
        """Run the neural architecture search experiment with multiple seeds"""
        print(f"Starting neural architecture search with {self.optimizer.__class__.__name__}")
        print(f"Surrogate model: {self.surrogate.__class__.__name__}")
        print(f"Number of runs: {self.num_runs} with seeds {self.seeds}")
        print(f"Training epochs per model: {self.surrogate.num_epochs}")
        print(f"Using the same population across all runs, only the neural network weights change")
        
        start_time = time.time()
        self.configure_optimizer()
        
        # Run evaluations for each seed
        for i, seed in enumerate(self.seeds):
            run_result = self.run_single_run(i, seed)
            self.run_results.append(run_result)
            
            # Update overall best if this run is better
            if run_result['best_fitness'] > self.best_fitness:
                self.best_fitness = run_result['best_fitness']
                self.best_individual = run_result['best_individual']
        
        # Report final results across all runs
        total_elapsed = time.time() - start_time
        best_perplexity = -self.best_fitness
        avg_perplexity = np.mean([result['best_perplexity'] for result in self.run_results])
        std_perplexity = np.std([result['best_perplexity'] for result in self.run_results])
        
        print("\nNeural Architecture Search Complete")
        print(f"Total runs: {self.num_runs}")
        print(f"Average validation perplexity: {avg_perplexity:.2f} ± {std_perplexity:.2f}")
        print(f"Best validation perplexity: {best_perplexity:.2f}")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Best architecture: {self.best_individual}")
        
        return self.best_individual