import os
import sys
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# For loggers imports
try:
    from logs.basic_logger import Logger
    from logs.opt_logger import Opt_Logger
except ImportError:
    print("Could not import datasets directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'logs'))
    from basic_logger import Logger
    from opt_logger import Opt_Logger

# For datasets imports
try:
    from datasets.ptb import get_ptb_dataset
    # from datasets.wt2 import get_wt2_dataset
    # from datasets.wt103 import *
except ImportError:
    print("Could not import datasets directly. Trying alternative import method...")
    # Alternative import if datasets is not a proper package
    sys.path.append(os.path.join(parent_dir, 'datasets'))
    from ptb import get_ptb_dataset
    # from wt2 import get_wt2_dataset
    # from wt103 import *

# For surrogates imports
try:
    # from surrogates.mintrain import MinTrainSurrogate
    from surrogates.mintrain_surrogate import SimplifiedMinTrainSurrogate
    # from surrogates.full_train import FullTrainSurrogate
    # from surrogates.random_forest import RandomForestSurrogate
except ImportError:
    print("Could not import surrogates directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'surrogates'))
    # from mintrain import MinTrainSurrogate
    from mintrain_surrogate import SimplifiedMinTrainSurrogate
    # from full_train import FullTrainSurrogate
    # from random_forest import RandomForestSurrogate

# For optimizers imports
try:
    from optimizers.ga import GeneticAlgorithm
    from optimizers.ge import GrammaticalEvolution
    # from optimizers.ga_cell import CellBasedGeneticAlgorithm
    # from optimizers.ge_cell import CellBasedGrammaticalEvolution
except ImportError:
    print("Could not import optimizers directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'optimizers'))
    from ga import GeneticAlgorithm
    from ge import GrammaticalEvolution
    # from ga_cell import CellBasedGeneticAlgorithm
    # from ge_cell import CellBasedGrammaticalEvolution

# For evaluators imports
try:
    from evaluators.evaluate import Evaluator
    # from evaluators.surrogate_evaluator import SurrEvaluator
except ImportError:
    print("Could not import evaluators directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'evaluators'))
    from evaluate import Evaluator
    # from surrogate_evaluator import SurrEvaluator

# For config imports
try:
    from configs.config_loader import ConfigLoader
except ImportError:
    print("Could not import config_loader directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'configs'))
    from config_loader import ConfigLoader

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run neural architecture search experiments")
    parser.add_argument('--config', type=str, default='default.json',
                        help='Configuration file name (default: default.json)')
    return parser.parse_args()

def main():
    """Run the neural architecture search experiment"""
    # Parse command line arguments and load config
    args = parse_arguments()
    config_loader = ConfigLoader(os.path.join(parent_dir, 'configs'))
    try:
        config = config_loader.load_config(args.config)
        print(f"Using configuration from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found. Using default configuration.")
        config = config_loader.load_config('default.json')

    # Set up objects
    supported_datasets = {
        'ptb': get_ptb_dataset(seq_length=config["dataset"]["seq_length"], batch_size=config["dataset"]["batch_size"]),
        # 'wt2': get_wt2_dataset(seq_length=config["dataset"]["seq_length"], batch_size=config["dataset"]["batch_size"]),
    }
    dataset = supported_datasets[config["dataset"]["name"]]

    supported_surrogates = {
        # 'base': FullTrainSurrogate(dataset, max_epochs=50, batch_size=128, patience=5, verbose=1),
        # 'mt': MinTrainSurrogate(dataset, dataset_name=config["dataset"]["name"], num_epochs=config["surrogate"]["num_epochs"], batch_size=config["surrogate"]["batch_size"], verbose=config["surrogate"]["verbose"]),
        'smt': SimplifiedMinTrainSurrogate(dataset, num_epochs=config["surrogate"]["num_epochs"], batch_size=config["surrogate"]["batch_size"], verbose=config["surrogate"]["verbose"])
        # 'rf': RandomForestSurrogate(dataset, initial_models=30, train_epochs=5, retrain_interval=20, verbose=1),
    }
    surrogate = supported_surrogates[config["surrogate"]["name"]]
    opt_logger = Opt_Logger(config=config)
    supported_optimizers = {
        'ga': GeneticAlgorithm(surrogate, pop_size=config["optimizer"]["pop_size"], generations=config["optimizer"]["generations"], mutation_rate=config["optimizer"]["mutation_rate"], crossover_rate=config["optimizer"]["crossover_rate"], seed=config["optimizer"]["seed"], logger=opt_logger),
        'ge': GrammaticalEvolution(surrogate, pop_size=config["optimizer"]["pop_size"], generations=config["optimizer"]["generations"], mutation_rate=config["optimizer"]["mutation_rate"], crossover_rate=config["optimizer"]["crossover_rate"], seed=config["optimizer"]["seed"], logger=opt_logger),
        # 'cell_ga': CellBasedGeneticAlgorithm(), # Not tested
        # 'cell_ge': CellBasedGrammaticalEvolution(), # Not implimented fully
    }
    optimizer = supported_optimizers[config["optimizer"]["name"]]
    logger = Logger(config)
    supported_evaluators = {
        'base': Evaluator(optimizer, dataset, max_runs=config["evaluator"]["num_runs"], log_interval=config["evaluator"]["log_interval"], full_runs=config["evaluator"]["full_runs"], logger=logger),
        # 'surrogate': SurrEvaluator(optimizer, num_runs=config["evaluator"]["num_runs"], log_interval=config["evaluator"]["log_interval"], starter_seed=config["evaluator"]["starter_seed"])
    }
    evaluator = supported_evaluators[config["evaluator"]["name"]]

    # Run experiment
    print(f"Running NAS experiment with {config['dataset']['name']} dataset using {config['surrogate']['name']} surrogate and {config['optimizer']['name']} optimizer")
    best_individual,  = evaluator.run(config["evaluator"]["starter_seed"])
    return best_individual

if __name__ == "__main__":
    main()