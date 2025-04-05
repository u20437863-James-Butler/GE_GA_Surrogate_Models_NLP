from datasets.ptb import get_ptb_dataset
from datasets.wt2 import get_wt2_dataset
# from datasets.wt103 import *
from surrogates.mintrain import MinTrainSurrogate
from surrogates.full_train import FullTrainSurrogate
from surrogates.random_forest import RandomForestSurrogate
from optimizers.ga import GeneticAlgorithm
from optimizers.ge import GrammaticalEvolution
from optimizers.ga_cell import CellBasedGeneticAlgorithm
from optimizers.ge_cell import CellBasedGrammaticalEvolution
from evaluators.evaluate import Evaluator, run_experiment
from evaluators.surrogate_evaluator import SurrEvaluator

supported_datasets = {
    'ptb': get_ptb_dataset(seq_length=35, batch_size=20),
    'wt2': get_wt2_dataset(seq_length=35, batch_size=20),
}

dataset_name = 'ptb'
dataset = supported_datasets[dataset_name]

supported_surrogates = {
    'base': FullTrainSurrogate(dataset, max_epochs=50, batch_size=128, patience=5, verbose=1),
    'mt': MinTrainSurrogate(dataset, num_epochs=50, batch_size=128, verbose=1),
    'rf': RandomForestSurrogate(dataset, initial_models=30, train_epochs=5, retrain_interval=20, verbose=1),
}

surrogate_name = 'mt'
surrogate = supported_surrogates[surrogate_name]

supported_optimizers = {
    'ga': GeneticAlgorithm(surrogate, pop_size=20, generations=5, mutation_rate=0.2, crossover_rate=0.8),
    'ge': GrammaticalEvolution(),
    'cell_ga': CellBasedGeneticAlgorithm(),
    'cell_ge': CellBasedGrammaticalEvolution(),
}

optimizer_name = 'ga'
optimizer = supported_optimizers[optimizer_name]

supported_evaluators = {
    'base': Evaluator(optimizer, max_evaluations=5, log_interval=5),
    'surrogate': SurrEvaluator(optimizer, max_evaluations=5, log_interval=5)
}
evaluator = supported_evaluators['surrogate']

def main():
    """Run the neural architecture search experiment"""
    print(f"Running NAS experiment with {dataset_name} dataset using {surrogate_name} surrogate and {optimizer_name} optimizer")
    
    # Run the experiment
    best_individual = evaluator.run()
    
    # Print the best architecture found
    print("\nBest Architecture Found:")
    print(best_individual)
    
    # Convert fitness back to perplexity
    best_perplexity = -best_individual.fitness
    print(f"Validation Perplexity: {best_perplexity:.2f}")
    
    # Return best individual for further analysis
    return best_individual

if __name__ == "__main__":
    main()