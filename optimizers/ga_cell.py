import random
import numpy as np
from optimizers.optimizer import Optimizer
from optimizers.ga_cell_Individual import CellBasedIndividual

class CellBasedGeneticAlgorithm(Optimizer):
    def __init__(self, surrogate, pop_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7, seed=None):
        """
        Initialize the Cell-Based Genetic Algorithm with a surrogate model for fitness evaluation.
        
        Args:
            surrogate: Surrogate instance for model evaluation
            pop_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
            seed: Random seed for reproducibility
        """
        self.surrogate = surrogate
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        
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
            
        self.population = [CellBasedIndividual(seed=self.seed+i if self.seed else None) for i in range(self.pop_size)]
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
        log_filename = f"generation_{self.current_generation}"
        fitness_scores = self.surrogate.evaluate_population(self.population, base_log_filename=log_filename)
        
        # Update best individual if needed
        for i, individual in enumerate(self.population):
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual.copy()
                
        return fitness_scores
    
    def select_parents(self):
        """
        Tournament selection - select the best individual from a random sample.
        """
        tournament_size = max(2, self.pop_size // 5)  # Adjust tournament size based on population
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
            return parent1.copy()
        
        # Create new child with shared seed for weight consistency
        child_seed = random.randint(0, 2**32 - 1)
        
        # Create new child
        child = CellBasedIndividual(seed=child_seed)
        
        # Crossover architectural parameters
        child.num_cells = random.choice([parent1.num_cells, parent2.num_cells])
        child.units = random.choice([parent1.units, parent2.units])
        child.dropout = random.choice([parent1.dropout, parent2.dropout])
        
        # Crossover cell structure
        child_cell = parent1.cell.copy() if random.random() < 0.5 else parent2.cell.copy()
        
        # If parents have different cell structures, we might perform more complex crossover
        if parent1.cell.num_nodes == parent2.cell.num_nodes:
            # Same size cells, can do element-wise mixing
            for i in range(child_cell.num_nodes):
                # Mix operations
                child_cell.operations[i] = random.choice([parent1.cell.operations[i], parent2.cell.operations[i]])
            
            # Mix connections while maintaining DAG property
            for i in range(child_cell.num_nodes):
                for j in range(i+1, child_cell.num_nodes):
                    child_cell.connections[i, j] = random.choice([
                        parent1.cell.connections[i, j],
                        parent2.cell.connections[i, j]
                    ])
        
        # Assign cell to child
        child.cell = child_cell
        
        # Ensure all nodes have at least one incoming connection (except node 0)
        for j in range(1, child.cell.num_nodes):
            if not np.any(child.cell.connections[:, j]):
                i = random.randint(0, j-1)
                child.cell.connections[i, j] = True
                
        return child
    
    def mutate(self, individual):
        """
        Randomly mutate genes of an individual.
        
        Args:
            individual: Individual to mutate
        """
        # Mutate architectural parameters
        if random.random() < self.mutation_rate:
            individual.num_cells = random.randint(1, individual.caps["num_cells"])
        if random.random() < self.mutation_rate:
            individual.units = random.randint(10, individual.caps["units"])
        if random.random() < self.mutation_rate:
            individual.dropout = random.uniform(0, individual.caps["dropout"])
        
        # Mutate cell structure
        individual.cell.mutate(self.mutation_rate)
    
    def evolve(self):
        """
        Run the genetic algorithm evolution process.
        
        Returns:
            Individual: The best individual found
        """
        print(f"Starting evolution with population size: {self.pop_size}, generations: {self.generations}")
        
        for gen in range(self.generations):
            self.current_generation = gen
            print(f"\nGeneration {gen+1}/{self.generations}")
            
            # Evaluate current population
            self.evaluate_population()
            
            # Create new population through selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep the best individual
            new_population.append(self.best_individual.copy())
            
            # Generate rest of population
            while len(new_population) < self.pop_size:
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
            print(f"Best fitness: {-self.surrogate.best_perplexity:.2f} (perplexity: {self.surrogate.best_perplexity:.2f})")
        
        # Final evaluation
        self.current_generation = self.generations
        print("\nFinal evaluation")
        self.evaluate_population()
        
        print(f"\nEvolution complete!")
        print(f"Best perplexity: {self.surrogate.best_perplexity:.4f}")
        print(f"Best architecture: {self.best_individual}")
        
        return self.best_individual