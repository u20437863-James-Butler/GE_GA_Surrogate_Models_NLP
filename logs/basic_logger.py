import os
import csv
import time
from datetime import datetime

class Logger:
    """
    Simple logger for neural architecture search experiments.
    Records experiment results to CSV files in the logs directory.
    """
    
    def __init__(self, config):
        self.config = config
        
        self.logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dataset_name = config['dataset']['name']
        optimizer_name = config['optimizer']['name']
        surrogate_name = config['surrogate']['name']
        self.base_filename = f"{dataset_name}_{optimizer_name}_{surrogate_name}_{self.timestamp}"
        
        self.experiment_log_path = os.path.join(self.logs_dir, f"{self.base_filename}_experiment.csv")
        self.initialize_experiment_log()
        
        print(f"Logging experiment results to: {self.experiment_log_path}")
    
    def initialize_experiment_log(self):
        """Initialize the main experiment log with configuration details"""
        with open(self.experiment_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Experiment Configuration'])
            writer.writerow(['Parameter', 'Value'])
            
            writer.writerow(['Dataset', self.config['dataset']['name']])
            writer.writerow(['Sequence Length', self.config['dataset']['seq_length']])
            writer.writerow(['Batch Size', self.config['dataset']['batch_size']])
            
            writer.writerow(['Surrogate', self.config['surrogate']['name']])
            writer.writerow(['Surrogate Epochs', self.config['surrogate']['num_epochs']])
            writer.writerow(['Surrogate Batch Size', self.config['surrogate']['batch_size']])
            
            writer.writerow(['Optimizer', self.config['optimizer']['name']])
            writer.writerow(['Population Size', self.config['optimizer']['pop_size']])
            writer.writerow(['Generations', self.config['optimizer']['generations']])
            writer.writerow(['Mutation Rate', self.config['optimizer']['mutation_rate']])
            writer.writerow(['Crossover Rate', self.config['optimizer']['crossover_rate']])
            writer.writerow(['Optimizer Seed', self.config['optimizer']['seed']])
            
            writer.writerow(['Number of Runs', self.config['evaluator']['num_runs']])
            writer.writerow(['Full Training Epochs', self.config['evaluator']['full_runs']])
            writer.writerow(['Starter Seed', self.config['evaluator']['starter_seed']])
            
            writer.writerow(['Timestamp', self.timestamp])
            writer.writerow([])
            
            # NEW header: we log based on TEST perplexity now
            writer.writerow(['Run', 'Evo Fitness', 'Full Training Test Perplexity', 'Time (s)'])
    
    def log_run(self, run_number, result):
        """Log results for a single run"""
        run_log_path = os.path.join(self.logs_dir, f"{self.base_filename}_run_{run_number}.csv")
        
        with open(run_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow(['Run Number', run_number])
            writer.writerow(['Evolutionary Fitness', result['fitness']])
            
            writer.writerow(['Architecture', str(result['individual'])])
            writer.writerow([])
            writer.writerow(['Full Training Results'])
            
            full_results = result['full_training_result']
            for key, value in full_results.items():
                writer.writerow([key, value])
        
        # When writing to experiment log, use test perplexity, not validation
        with open(self.experiment_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_number,
                f"{result['fitness']:.4f}",
                f"{result['full_training_result']['test_perplexity']:.4f}",
                f"{result.get('time', 0):.1f}"
            ])
        
        print(f"Logged run {run_number} results to: {run_log_path}")
    
    def log_final_results(self, best_individual, all_results, total_time):
        """Log final results of the experiment"""
        with open(self.experiment_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([])
            writer.writerow(['Final Results'])
            writer.writerow(['Total Experiment Time (s)', f"{total_time:.1f}"])
            writer.writerow(['Best Fitness', f"{best_individual.fitness:.4f}"])
            writer.writerow(['Best Architecture', str(best_individual)])
            
            avg_fitness = sum(r['fitness'] for r in all_results) / len(all_results)
            avg_perplexity = sum(r['full_training_result']['test_perplexity'] for r in all_results) / len(all_results)
            
            writer.writerow(['Average Fitness', f"{avg_fitness:.4f}"])
            writer.writerow(['Average Test Perplexity', f"{avg_perplexity:.4f}"])
        
        print(f"Logged final experiment results to: {self.experiment_log_path}")
