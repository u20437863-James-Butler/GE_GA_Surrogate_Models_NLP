import random
import numpy as np
from optimizers.optimizer import Optimizer
from optimizers.ga_individual import GA_Individual

class GeneticAlgorithm(Optimizer):
    def __init__(self, surrogate, pop_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7, seed=None):
        """
        Initialize the Genetic Algorithm with a surrogate model for fitness evaluation.
        
        Args:
            surrogate: Surrogate instance for model evaluation
            pop_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
        """
        self.surrogate = surrogate
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed

        # Track for early stopping
        self.patience = 3
        self.gens_since_last_improvement = 0
        self.current_generation = 0
        
        # Generate random initial population
        self.population = self.generate_population(seed=self.seed)
        
        # Track best individual and fitness
        self.best_individual = None
        self.best_fitness = float('-inf')  # Higher fitness is better
        
        # Get input shape and output dim from surrogate
        self.input_shape = surrogate.input_shape
        self.output_dim = surrogate.output_dim
    
    def generate_population(self, seed=None):
        """
        Generate a new population with an optional seed for reproducibility.
        
        Args:
            seed: Random seed for population generation
            
        Returns:
            list: New population of individuals
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.population = [GA_Individual(seed=self.seed+i) for i in range(self.pop_size)]
        return self.population
        
    def evaluate_only(self, population=None, seed=None, base_log_filename=None):
        """
        Evaluate the population without evolving it.
        Useful for evaluating initial populations or for benchmarking.
        
        Args:
            population: Optional pre-generated population to evaluate
            seed: Optional seed for weight initialization
            base_log_filename: Base filename for logging
            
        Returns:
            list: List of fitness scores
        """
        # Use provided population if given, otherwise use the existing one
        eval_population = population if population is not None else self.population
        
        if seed is not None:
            # Update seeds for consistent weight initialization
            for individual in eval_population:
                individual.seed = seed
        
        # Use the surrogate to evaluate all individuals
        fitness_scores = self.surrogate.evaluate_population(eval_population, seed=seed, base_log_filename=base_log_filename)
        
        # Update best individual if needed
        for individual in eval_population:
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual.copy()
                
        return fitness_scores
    
    def evaluate_population(self):
        """
        Evaluate the entire population using the surrogate model.
        """
        # Use the surrogate to evaluate all individuals
        fitness_scores = self.surrogate.evaluate_population(self.population)
        # Update best individual if needed
        for i, individual in enumerate(self.population):
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual.copy()
                self.best_individual.fitness = self.best_fitness
                self.gens_since_last_improvement = 0
            else:
                self.gens_since_last_improvement += 1
        return fitness_scores
    
    def select_parents(self):
        """
        Tournament selection - select the best individual from a random sample.
        """
        tournament_size = max(2, self.pop_size // 5)
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create a child.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Individual: New child individual
        """
        if random.random() > self.crossover_rate:
            return random.choice([parent1.copy(), parent2.copy()])
        
        # Create new child with shared seed for weight consistency
        child_seed = random.choice([parent1.seed, parent2.seed])
        
        # Choose layer count from either parent
        layer_counts = random.choice([parent1.layer_counts, parent2.layer_counts])
        
        # Build units and activations from each parent with proper padding
        units = []
        activations = []
        
        for i in range(layer_counts):
            # Select parent to get unit from
            parent_for_unit = random.choice([parent1, parent2])
            # Select parent to get activation from
            parent_for_activation = random.choice([parent1, parent2])
            
            # Get unit with padding if needed
            if i < len(parent_for_unit.units):
                units.append(parent_for_unit.units[i])
            else:
                # Padding with default value if parent doesn't have enough layers
                units.append(16)  # Default unit size
                
            # Get activation with padding if needed
            if i < len(parent_for_activation.activations):
                activations.append(parent_for_activation.activations[i])
            else:
                # Padding with default value if parent doesn't have enough layers
                activations.append('tanh')  # Default activation
        
        # Ensure last layer has return_sequences=False if there are multiple layers
        return_sequences = [True] * (layer_counts - 1) + [False]
        
        dropout = random.choice([parent1.dropout, parent2.dropout])
        
        # Create child
        child = GA_Individual(
            seed=child_seed,
            layer_counts=layer_counts,
            units=units,
            activations=activations,
            dropout=dropout
        )
        
        return child
    
    def mutate(self, individual):
        """
        Randomly mutate genes of an individual.
        
        Args:
            individual: Individual to mutate
        """
        # Each property has a chance to mutate based on mutation rate
        if random.random() < self.mutation_rate:
            # Select one random aspect to mutate
            mutation_type = random.randint(0, 3)
            
            if mutation_type == 0:
                # Mutate layer count
                individual.layer_counts = random.randint(1, 3)
                # Adjust lengths of other properties accordingly
                individual.units = individual.units[:individual.layer_counts]
                individual.activations = individual.activations[:individual.layer_counts]
                
                # Add new layers if needed
                while len(individual.units) < individual.layer_counts:
                    individual.units.append(random.choice([16,32,64,128]))
                while len(individual.activations) < individual.layer_counts:
                    individual.activations.append(random.choice(['relu', 'tanh', 'sigmoid']))
                    
                # Ensure last layer has return_sequences=False if multiple layers
                individual.return_sequences = [True] * (individual.layer_counts - 1) + [False]
                
            elif mutation_type == 1:
                # Mutate one unit size
                layer_idx = random.randint(0, individual.layer_counts - 1)
                individual.units[layer_idx] = random.choice([16,32,64,128])
                
            elif mutation_type == 2:
                # Mutate one activation
                layer_idx = random.randint(0, individual.layer_counts - 1)
                individual.activations[layer_idx] = random.choice(['relu', 'tanh', 'sigmoid'])
                    
            elif mutation_type == 3:
                # Mutate dropout
                individual.dropout = random.randint(0, 9) * 0.1
    
    def evolve(self):
        """
        Run the genetic algorithm evolution process.
        
        Returns:
            Individual: The best individual found
        """
        print(f"Starting evolution with population size: {self.pop_size}, generations: {self.generations}")
        
        early_stopping_flag = False

        for gen in range(self.generations):
            self.current_generation = gen
            print(f"\nGeneration {gen+1}/{self.generations}")
            
            # Evaluate current population
            self.evaluate_population()
            
            # Check early stopping criteria
            if self.gens_since_last_improvement == self.patience:
                early_stopping_flag = True
                break

            # Create new population through selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(self.best_individual.copy())
            
            # Generate rest of population
            while len(new_population) < self.pop_size:
                # Tournament selection
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                
                # Create child through crossover
                child = self.crossover(parent1, parent2)
                
                # Mutate child
                self.mutate(child)
                
                # Add to new population
                new_population.append(child)
            
            # Replace old population
            self.population = new_population
            
            # Print progress
            print(f"Best fitness: {-self.surrogate.best_perplexity:.5f} (perplexity: {self.surrogate.best_perplexity:.2f})")
        
        if early_stopping_flag:
            print("\nStopped early at generation:",self.current_generation+1)
        else:
            # Final evaluation
            self.current_generation = self.generations
            print("\nFinal evaluation")
            self.evaluate_population()
        
        print(f"\nEvolution complete!")
        print(f"Best perplexity: {self.surrogate.best_perplexity:.4f}")
        print(f"Best architecture: {self.best_individual}")
        print(f"Best fitness: {self.best_individual.fitness}")
        return self.best_individual