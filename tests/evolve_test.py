import os
import sys
import random
import numpy as np
import hashlib

# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import optimizers
try:
    from optimizers.ga import GeneticAlgorithm
    from optimizers.ge import GrammaticalEvolution
    from surrogates.mintrain_surrogate import SimplifiedMinTrainSurrogate
    from datasets.ptb import get_ptb_dataset
except ImportError:
    print("Could not import optimizers directly. Trying alternative import method...")
    sys.path.append(os.path.join(parent_dir, 'optimizers'))
    from ga import GeneticAlgorithm
    from ge import GrammaticalEvolution
    sys.path.append(os.path.join(parent_dir, 'surrogates'))
    from mintrain_surrogate import SimplifiedMinTrainSurrogate
    sys.path.append(os.path.join(parent_dir, 'datasets'))
    from ptb import get_ptb_dataset

class SimpleSurrogate:
    """A simple surrogate model that assigns fitness based on the sum of ASCII values in the ID."""
    
    def __init__(self):
        self.input_shape = (35,)  # Dummy input shape
        self.output_dim = 10000   # Dummy output dimension
        self.best_perplexity = float('inf')
        
    def evaluate_population(self, population, seed=None, base_log_filename=None):
        """Evaluate all individuals in the population."""
        fitness_scores = []
        
        for ind in population:
            # Calculate fitness based on ASCII sum of the individual's ID
            id_string = ind.getId()
            ascii_sum = sum(ord(c) for c in id_string)
            
            # Invert score to simulate perplexity (lower is better)
            perplexity = 1000.0 / (ascii_sum + 1)
            
            # Set negative perplexity as fitness (higher is better)
            ind.fitness = -perplexity
            fitness_scores.append(ind.fitness)
            
            # Update best perplexity if needed
            if perplexity < self.best_perplexity:
                self.best_perplexity = perplexity
                
            print(f"Individual {id_string}: ASCII sum = {ascii_sum}, Perplexity = {perplexity:.4f}, Fitness = {ind.fitness:.4f}")
        
        return fitness_scores

def test_ga_evolution():
    """Test the GA evolution process with the simple surrogate."""
    print("\n========== Testing Genetic Algorithm Evolution ==========")
    
    # Create simple surrogate
    # surrogate = SimpleSurrogate()
    surrogate = SimplifiedMinTrainSurrogate(get_ptb_dataset(seq_length=35, batch_size=20))
    # Create GA optimizer with small population and generations
    ga = GeneticAlgorithm(
        surrogate=surrogate,
        pop_size=3,
        generations=2,
        mutation_rate=0.3,
        crossover_rate=0.7,
        seed=42
    )
    
    # Generate initial population
    print("\nGenerating initial GA population...")
    population = ga.generate_population(seed=42)
    
    print("\nInitial population:")
    for i, ind in enumerate(population):
        print(f"Individual {i}: {ind}")
    
    # Run evolution
    print("\nRunning GA evolution...")
    best_individual = ga.evolve()
    
    print("\nEvolution completed!")
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness:.4f}")
    
    return best_individual

def test_ge_evolution():
    """Test the GE evolution process with the simple surrogate."""
    print("\n========== Testing Grammatical Evolution Evolution ==========")
    
    # Create simple surrogate
    # surrogate = SimpleSurrogate()
    surrogate = SimplifiedMinTrainSurrogate(get_ptb_dataset(seq_length=35, batch_size=20))


    # Create GE optimizer with small population and generations
    ge = GrammaticalEvolution(
        surrogate=surrogate,
        pop_size=3,
        generations=2,
        mutation_rate=0.3,
        crossover_rate=0.7,
        max_genotype_length=5,
        seed=42
    )
    
    # Generate initial population
    print("\nGenerating initial GE population...")
    population = ge.generate_population(seed=42)
    
    print("\nInitial population:")
    for i, ind in enumerate(population):
        print(f"Individual {i}: {ind}")
        # Initialize phenotypes
        phenotype = ge.genotype_to_phenotype(ind.genotype)
        ind.setPhenotype(phenotype)
    
    # Run evolution
    print("\nRunning GE evolution...")
    best_individual = ge.evolve()
    
    print("\nEvolution completed!")
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness:.4f}")
    
    return best_individual

def main():
    """Run evolution tests for both GA and GE."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test GA evolution
    # ga_best = test_ga_evolution()
    
    # Test GE evolution
    ge_best = test_ge_evolution()
    
    # Compare results
    print("\n========== Comparison of Results ==========")
    # print(f"GA Best Fitness: {ga_best.fitness:.4f}")
    print(f"GE Best Fitness: {ge_best.fitness:.4f}")
    
    # if ga_best.fitness > ge_best.fitness:
    #     print("GA performed better in this test.")
    # elif ge_best.fitness > ga_best.fitness:
    #     print("GE performed better in this test.")
    # else:
    #     print("Both algorithms performed equally.")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()