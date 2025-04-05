import random
import numpy as np
from optimizers.GA_Individual import Individual

class GeneticAlgorithm:
    def __init__(self, surrogate, pop_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7):
        """
        Initialize the Genetic Algorithm with a surrogate model for fitness evaluation.
        
        Args:
            surrogate: MinTrainSurrogate instance for model evaluation
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
        
        # Generate random initial population
        self.population = [Individual() for _ in range(pop_size)]
        
        # Track best individual and fitness
        self.best_individual = None
        self.best_fitness = float('-inf')  # Higher fitness is better
        
        # Get input shape and output dim from surrogate
        self.input_shape = surrogate.input_shape
        self.output_dim = surrogate.output_dim
        
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
        
        # Choose layer count from either parent
        layer_counts = random.choice([parent1.layer_counts, parent2.layer_counts])
        
        # Make sure arrays have proper length
        max_layers = max(parent1.layer_counts, parent2.layer_counts)
        
        # Extend parent arrays if needed for proper crossover
        p1_rnn_types = parent1.rnn_types + ['LSTM'] * (max_layers - len(parent1.rnn_types))
        p2_rnn_types = parent2.rnn_types + ['LSTM'] * (max_layers - len(parent2.rnn_types))
        
        p1_units = parent1.units + [64] * (max_layers - len(parent1.units))
        p2_units = parent2.units + [64] * (max_layers - len(parent2.units))
        
        p1_activations = parent1.activations + ['tanh'] * (max_layers - len(parent1.activations))
        p2_activations = parent2.activations + ['tanh'] * (max_layers - len(parent2.activations))
        
        p1_return_sequences = parent1.return_sequences + [False] * (max_layers - len(parent1.return_sequences))
        p2_return_sequences = parent2.return_sequences + [False] * (max_layers - len(parent2.return_sequences))
        
        # Create crossover arrays
        rnn_types = [random.choice([p1, p2]) for p1, p2 in zip(p1_rnn_types, p2_rnn_types)][:layer_counts]
        units = [random.choice([p1, p2]) for p1, p2 in zip(p1_units, p2_units)][:layer_counts]
        activations = [random.choice([p1, p2]) for p1, p2 in zip(p1_activations, p2_activations)][:layer_counts]
        return_sequences = [random.choice([p1, p2]) for p1, p2 in zip(p1_return_sequences, p2_return_sequences)][:layer_counts]
        
        # Ensure last layer has return_sequences=False if there are multiple layers
        if layer_counts > 1:
            return_sequences[-1] = False
            
        dropout = random.choice([parent1.dropout, parent2.dropout])
        
        # Create child
        child = Individual(
            seed=child_seed,
            layer_counts=layer_counts,
            rnn_types=rnn_types,
            units=units,
            activations=activations,
            return_sequences=return_sequences,
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
            mutation_type = random.randint(0, 5)
            
            if mutation_type == 0:
                # Mutate layer count
                individual.layer_counts = random.randint(1, 3)
                # Adjust lengths of other properties accordingly
                individual.rnn_types = individual.rnn_types[:individual.layer_counts]
                individual.units = individual.units[:individual.layer_counts]
                individual.activations = individual.activations[:individual.layer_counts]
                individual.return_sequences = individual.return_sequences[:individual.layer_counts]
                
                # Add new layers if needed
                while len(individual.rnn_types) < individual.layer_counts:
                    individual.rnn_types.append(random.choice(['SimpleRNN', 'LSTM', 'GRU']))
                while len(individual.units) < individual.layer_counts:
                    individual.units.append(random.randint(1, 10) * 10)
                while len(individual.activations) < individual.layer_counts:
                    individual.activations.append(random.choice(['relu', 'tanh', 'sigmoid']))
                while len(individual.return_sequences) < individual.layer_counts:
                    individual.return_sequences.append(True)
                    
                # Ensure last layer has return_sequences=False if multiple layers
                if individual.layer_counts > 1:
                    individual.return_sequences[-1] = False
                    
            elif mutation_type == 1:
                # Mutate one RNN type
                layer_idx = random.randint(0, individual.layer_counts - 1)
                individual.rnn_types[layer_idx] = random.choice(['SimpleRNN', 'LSTM', 'GRU'])
                
            elif mutation_type == 2:
                # Mutate one unit size
                layer_idx = random.randint(0, individual.layer_counts - 1)
                individual.units[layer_idx] = random.randint(1, 10) * 10
                
            elif mutation_type == 3:
                # Mutate one activation
                layer_idx = random.randint(0, individual.layer_counts - 1)
                individual.activations[layer_idx] = random.choice(['relu', 'tanh', 'sigmoid'])
                
            elif mutation_type == 4:
                # Mutate one return_sequences (but not the last one in multi-layer models)
                if individual.layer_counts > 1:
                    layer_idx = random.randint(0, individual.layer_counts - 2)
                    individual.return_sequences[layer_idx] = not individual.return_sequences[layer_idx]
                    
            elif mutation_type == 5:
                # Mutate dropout
                individual.dropout = random.uniform(0, 0.5)
    
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