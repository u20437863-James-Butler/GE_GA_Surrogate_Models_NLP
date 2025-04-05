import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
from datetime import datetime

class SurrEvaluator:
    """
    Evaluator class to run neural architecture search experiments.
    
    This class takes a surrogate model and dataset, then runs an 
    evolutionary optimizer to find optimal neural architectures.
    """
    
    def __init__(self, optimizer, max_evaluations=100, log_interval=10, num_runs=1, seeds=None):
        """
        Initialize the evaluator.
        
        Args:
            optimizer: An evolutionary algorithm optimizer
            max_evaluations: Maximum number of evaluations to perform
            log_interval: How often to log progress
            num_runs: Number of runs with different seeds to perform
            seeds: List of seeds to use for runs (generated if None)
        """
        self.optimizer = optimizer
        self.surrogate = optimizer.surrogate
        self.max_evaluations = max_evaluations
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
        self.fitness_history = []
        self.best_fitness_history = []
        self.evaluation_count = 0
        self.start_time = None
        
        # Results across multiple runs
        self.run_results = []
        
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
    
    def run_single_evaluation(self, run_index, seed):
        """
        Run a single evaluation with a specific seed.
        
        Args:
            run_index: Index of the current run
            seed: Seed for weight initialization and population generation
            
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
        
        # Create population with seed
        self.optimizer.surrogate = new_surrogate  # Point optimizer to new surrogate
        population = self.optimizer.generate_population(seed=seed)
        
        # Evaluate population
        start_time = time.time()
        fitness_scores = self.optimizer.evaluate_only(seed=seed, base_log_filename=run_name)
        elapsed = time.time() - start_time
        
        # Get best individual from this run
        best_individual = self.optimizer.best_individual
        best_fitness = best_individual.fitness if best_individual else float('-inf')
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
        
        self.start_time = time.time()
        self.configure_optimizer()
        
        # Run evaluations for each seed
        for i, seed in enumerate(self.seeds):
            run_result = self.run_single_evaluation(i, seed)
            self.run_results.append(run_result)
            
            # Update overall best if this run is better
            if run_result['best_fitness'] > self.best_fitness:
                self.best_fitness = run_result['best_fitness']
                self.best_individual = run_result['best_individual']
        
        # Report final results across all runs
        total_elapsed = time.time() - self.start_time
        best_perplexity = -self.best_fitness
        avg_perplexity = np.mean([result['best_perplexity'] for result in self.run_results])
        std_perplexity = np.std([result['best_perplexity'] for result in self.run_results])
        
        print("\nNeural Architecture Search Complete")
        print(f"Total runs: {self.num_runs}")
        print(f"Average validation perplexity: {avg_perplexity:.2f} ± {std_perplexity:.2f}")
        print(f"Best validation perplexity: {best_perplexity:.2f}")
        print(f"Total time: {total_elapsed:.1f}s")
        print(f"Best architecture: {self.best_individual.__dict__ if self.best_individual else None}")
        
        # Plot results across runs
        self.plot_run_results()
        
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
        """Plot optimization progress for the best run"""
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
        
    def plot_run_results(self):
        """Plot results across multiple runs"""
        if not self.run_results:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract perplexities from each run
        seeds = [result['seed'] for result in self.run_results]
        perplexities = [result['best_perplexity'] for result in self.run_results]
        
        # Bar chart of perplexities by seed
        plt.bar(range(len(seeds)), perplexities)
        plt.xticks(range(len(seeds)), [f"Seed {seed}" for seed in seeds], rotation=45)
        plt.axhline(y=np.mean(perplexities), color='r', linestyle='-', label=f'Mean: {np.mean(perplexities):.2f}')
        
        plt.xlabel('Run')
        plt.ylabel('Validation Perplexity (lower is better)')
        plt.title('Performance Across Multiple Seeds')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y')
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/multi_run_results.png')
        plt.close()