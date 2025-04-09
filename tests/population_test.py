import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import required modules
try:
    from datasets.ptb import get_ptb_dataset
except ImportError:
    print("Could not import datasets directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'datasets'))
    from ptb import get_ptb_dataset

try:
    from surrogates.mintrain import MinTrainSurrogate
except ImportError:
    print("Could not import surrogates directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'surrogates'))
    from mintrain import MinTrainSurrogate

# Direct imports for optimizers
try:
    from optimizers.ga import GeneticAlgorithm
    from optimizers.ge import GrammaticalEvolution
except ImportError:
    print("Could not import optimizers directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'optimizers'))
    from ga import GeneticAlgorithm
    from ge import GrammaticalEvolution

def test_population_generation(optimizer, seed=42):
    """
    Test population generation functionality
    
    Args:
        optimizer: Optimizer instance (GA, GE, etc.)
        seed: Seed for reproducibility
    """
    print("\n==== Testing Population Generation ====")
    population = optimizer.generate_population(seed=seed)
    
    print(f"Population size: {len(population)}")
    print(f"First individual type: {type(population[0]).__name__}")
    
    # Print a sample of individuals
    print("\nSample of individuals:")
    for i in range(min(3, len(population))):
        print(f"Individual {i}: {str(population[i])}")
    
    return population

def test_model_building(population, input_shape, output_dim):
    """
    Test that models can be built from individuals
    
    Args:
        population: List of Individual instances
        input_shape: Input shape for the models
        output_dim: Output dimension for the models
    """
    print("\n==== Testing Model Building ====")
    
    models = []
    for i, individual in enumerate(population[:3]):  # Test just the first 3 individuals
        print(f"\nBuilding model {i} from individual")
        
        try:
            # Build model from the individual
            model = individual.build_model(input_shape, output_dim)
            
            # Print model summary
            print(f"Model {i} successfully built:")
            model.summary(print_fn=lambda x: print(f"  {x}"))
            
            models.append(model)
        except Exception as e:
            print(f"Error building model {i}: {str(e)}")
    
    return models

def test_model_training_inference(models, dataset):
    """
    Test that models can be trained and used for inference
    
    Args:
        models: List of Keras models
        dataset: Dataset for training and inference
    """
    print("\n==== Testing Model Training and Inference ====")
    
    for i, model in enumerate(models):
        print(f"\nTesting model {i}")
        
        try:
            # Get a small batch of data for testing
            x_train, y_train = dataset.get_train_batch(batch_size=32)
            
            # Train the model for 1 epoch
            print("Training model...")
            history = model.fit(
                x_train, y_train, 
                epochs=1, 
                batch_size=2048, 
                verbose=1
            )
            
            # Get validation data for inference
            x_val, y_val = dataset.get_valid_batch(batch_size=32)
            
            # Evaluate on validation data
            print("Evaluating model...")
            evaluation = model.evaluate(x_val, y_val, verbose=1)
            
            print(f"Validation loss: {evaluation[0]:.4f}, Validation accuracy: {evaluation[1]:.4f}")
            
            # Test inference
            print("Testing inference...")
            predictions = model.predict(x_val[:5], verbose=1)
            print(f"Prediction shape: {predictions.shape}")
            print(f"Prediction sample: {predictions[0][:5]}")  # Show first 5 probabilities of first sample
            
        except Exception as e:
            print(f"Error testing model {i}: {str(e)}")

def test_individual_evaluation(optimizer, population, surrogate):
    """
    Test that individuals can be evaluated using the surrogate
    
    Args:
        optimizer: Optimizer instance
        population: Population of individuals
        surrogate: Surrogate instance for evaluation
    """
    print("\n==== Testing Individual Evaluation ====")
    try:
        print("Evaluating population...")
        fitness_scores = optimizer.evaluate_only(population=population[:3], base_log_filename="test_evaluation")
        
        print("Evaluation results:")
        for i, ind in enumerate(population[:3]):
            print(f"Individual {i}: Fitness = {ind.fitness}")
            
        return True
    except Exception as e:
        print(f"Error evaluating individuals: {str(e)}")
        return False

def test_ga_optimizer(surrogate, seed=42):
    """
    Test the Genetic Algorithm optimizer
    
    Args:
        surrogate: Surrogate instance for evaluation
        seed: Random seed for reproducibility
    """
    print("\n========== Testing Genetic Algorithm ==========")
    
    # Create optimizer with a small population size for testing
    optimizer = GeneticAlgorithm(
        surrogate=surrogate,
        pop_size=5,  # Small population for testing
        generations=1,
        mutation_rate=0.2,
        crossover_rate=0.7
    )
    
    # Test population generation
    population = test_population_generation(optimizer, seed)
    
    if population:
        # Test model building
        input_shape = surrogate.input_shape
        output_dim = surrogate.output_dim
        models = test_model_building(population, input_shape, output_dim)
        
        # # Test model training and inference
        # if models:
        #     test_model_training_inference(models, surrogate._dataset)
        
        # Test individual evaluation
        test_individual_evaluation(optimizer, population, surrogate)
    
    print("\n========== Genetic Algorithm tests completed! ==========")

def test_ge_optimizer(surrogate, seed=42):
    """
    Test the Grammatical Evolution optimizer
    
    Args:
        surrogate: Surrogate instance for evaluation
        seed: Random seed for reproducibility
    """
    print("\n========== Testing Grammatical Evolution ==========")
    
    # Create optimizer with a small population size for testing
    optimizer = GrammaticalEvolution(
        surrogate=surrogate,
        pop_size=5,  # Small population for testing
        generations=1,
        mutation_rate=0.2,
        crossover_rate=0.7,
        max_genotype_length=10  # Set explicitly for testing
    )
    
    # Test population generation
    population = test_population_generation(optimizer, seed)
    
    if population:
        # Test model building
        input_shape = surrogate.input_shape
        output_dim = surrogate.output_dim
        
        # For GE, we need to first convert genotypes to phenotypes
        for individual in population[:3]:
            phenotype = optimizer.genotype_to_phenotype(individual.genotype)
            print(f"\nGenerated model code: {phenotype}")
            individual.setPhenotype(phenotype)
            print(individual.phenotype)
        
        # Test model building from phenotypes
        models = test_model_building(population, input_shape, output_dim)
        
        # Test model training and inference
        if models:
            test_model_training_inference(models, surrogate._dataset)
        
        # Test individual evaluation
        test_individual_evaluation(optimizer, population, surrogate)
    
    print("\n========== Grammatical Evolution tests completed! ==========")

def main():
    """Run tests for multiple optimizers"""
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Get dataset
    dataset_name = 'ptb'
    dataset = get_ptb_dataset(seq_length=35, batch_size=20)
    
    # Create surrogate
    surrogate = MinTrainSurrogate(dataset, dataset_name=dataset_name, num_epochs=1, batch_size=8192, verbose=1)
    
    # Test each optimizer
    # test_ga_optimizer(surrogate)
    test_ge_optimizer(surrogate)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()