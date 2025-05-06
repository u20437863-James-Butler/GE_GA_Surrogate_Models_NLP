import os
import csv
import time
import json
from datetime import datetime

class Opt_Logger:
    def __init__(self, config):
        """
        Initialize the optimization logger.
        
        Args:
            config: Configuration dictionary containing experiment parameters
        """
        self.config = config
        self.start_time = time.time()
        self.gen_start_time = self.start_time
        
        # Generate a unique directory name based on the config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimizer_name = config['optimizer']['name']
        dataset_name = config['dataset']['name']
        surrogate_name = config['surrogate']['name']
        
        # Create custom directory name
        self.dir_name = f"{dataset_name}_{optimizer_name}_{surrogate_name}_{timestamp}_optimizer"
        
        # Create logs/results directory structure if it doesn't exist
        self.log_dir = os.path.join('logs', 'results', self.dir_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set filename for CSV within the directory
        self.filename = os.path.join(self.log_dir, "generations.csv")
        
        # Initialize the CSV file with headers
        self.headers = [
            'generation', 
            'best_fitness', 
            'best_perplexity', 
            'best_architecture',
            'gen_runtime_sec', 
            'total_runtime_sec'
        ]
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        # Log the configuration
        self.log_config()
    
    def log_config(self):
        """Log the configuration as a separate JSON file in the same directory"""
        config_filename = os.path.join(self.log_dir, "config.json")
        with open(config_filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def start_generation(self):
        """Mark the start time of a new generation"""
        self.gen_start_time = time.time()
    
    def log_generation(self, generation, best_fitness, best_perplexity, best_architecture):
        """
        Log information about the current generation.
        
        Args:
            generation: Current generation number
            best_fitness: Best fitness value found
            best_perplexity: Best perplexity value found
            best_architecture: String representation of the best architecture
        """
        gen_runtime = time.time() - self.gen_start_time
        total_runtime = time.time() - self.start_time
        
        row = [
            generation,
            best_fitness,
            best_perplexity,
            best_architecture,
            gen_runtime,
            total_runtime
        ]
        
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        return gen_runtime, total_runtime
    
    def log_final_results(self, best_individual, best_perplexity, total_generations):
        """
        Log the final results of the optimization.
        
        Args:
            best_individual: The best individual found
            best_perplexity: The best perplexity value found
            total_generations: The total number of generations run
        """
        final_runtime = time.time() - self.start_time
        
        summary_filename = os.path.join(self.log_dir, "summary.txt")
        with open(summary_filename, 'w') as f:
            f.write(f"Optimization Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Total generations: {total_generations}\n")
            f.write(f"Total runtime: {final_runtime:.2f} seconds\n")
            f.write(f"Best perplexity: {best_perplexity:.4f}\n")
            f.write(f"Best architecture: {str(best_individual)}\n")