import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# For datasets imports
try:
    from datasets.ptb import get_ptb_dataset
    from datasets.wt2 import get_wt2_dataset
    # from datasets.wt103 import *
except ImportError:
    print("Could not import datasets directly. Trying alternative import method...")
    # Alternative import if datasets is not a proper package
    sys.path.append(os.path.join(parent_dir, 'datasets'))
    from ptb import get_ptb_dataset
    from wt2 import get_wt2_dataset
    # from wt103 import *

# For surrogates imports
try:
    from surrogates.mintrain import MinTrainSurrogate
    # from surrogates.full_train import FullTrainSurrogate
    from surrogates.random_forest import RandomForestSurrogate
except ImportError:
    print("Could not import surrogates directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'surrogates'))
    from mintrain import MinTrainSurrogate
    # from full_train import FullTrainSurrogate
    # from random_forest import RandomForestSurrogate

# For optimizers imports
try:
    from optimizers.ga import GeneticAlgorithm
    # from optimizers.ge import GrammaticalEvolution
    # from optimizers.ga_cell import CellBasedGeneticAlgorithm
    # from optimizers.ge_cell import CellBasedGrammaticalEvolution
except ImportError:
    print("Could not import optimizers directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'optimizers'))
    from ga import GeneticAlgorithm
    # from ge import GrammaticalEvolution
    # from ga_cell import CellBasedGeneticAlgorithm
    # from ge_cell import CellBasedGrammaticalEvolution

# For evaluators imports
try:
    # from evaluators.evaluate import Evaluator, run_experiment
    from evaluators.surrogate_evaluator import SurrEvaluator
except ImportError:
    print("Could not import evaluators directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'evaluators'))
    # from evaluate import Evaluator, run_experiment
    from surrogate_evaluator import SurrEvaluator

supported_datasets = {
    'ptb': get_ptb_dataset(seq_length=35, batch_size=20),
    'wt2': get_wt2_dataset(seq_length=35, batch_size=20),
}

dataset_name = 'ptb'
dataset = supported_datasets[dataset_name]

supported_surrogates = {
    # 'base': FullTrainSurrogate(dataset, max_epochs=50, batch_size=128, patience=5, verbose=1),
    'mt': MinTrainSurrogate(dataset, dataset_name=dataset_name, num_epochs=1, batch_size=2048, verbose=1),
    # 'rf': RandomForestSurrogate(dataset, initial_models=30, train_epochs=5, retrain_interval=20, verbose=1),
}

surrogate_name = 'mt'
surrogate = supported_surrogates[surrogate_name]

supported_optimizers = {
    'ga': GeneticAlgorithm(surrogate, pop_size=1, generations=5, mutation_rate=0.2, crossover_rate=0.8),
    # 'ge': GrammaticalEvolution(),
    # 'cell_ga': CellBasedGeneticAlgorithm(),
    # 'cell_ge': CellBasedGrammaticalEvolution(),
}

optimizer_name = 'ga'
optimizer = supported_optimizers[optimizer_name]

supported_evaluators = {
    # 'base': Evaluator(optimizer, max_evaluations=5, log_interval=5),
    'surrogate': SurrEvaluator(optimizer, num_runs=1, log_interval=1)
}
evaluator = supported_evaluators['surrogate']

def main():
    """Run the neural architecture search experiment"""
    print(f"Running NAS experiment with {dataset_name} dataset using {surrogate_name} surrogate and {optimizer_name} optimizer")
    
    # Run the experiment
    best_individual = evaluator.run()
    
    # Return best individual for further analysis
    return best_individual

if __name__ == "__main__":
    main()