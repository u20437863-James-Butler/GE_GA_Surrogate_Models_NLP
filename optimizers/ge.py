import random
import numpy as np
from tensorflow import keras
from keras import Sequential, layers
from optimizers.optimizer import Optimizer
from optimizers.ge_individual import GE_Individual

class GrammaticalEvolution(Optimizer):
    def __init__(self, surrogate, pop_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7, max_genotype_length=8, seed=None):
        """
        Initialize the Grammatical Evolution with a surrogate model for fitness evaluation.
        
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

        self.population = self.generate_population(seed=self.seed)

        # Track best individual and fitness
        self.best_individual = None
        self.best_fitness = float('-inf')  # Higher fitness is better

        self.grammar = {
            '<rnn-model>': ["<rnn-layers>\nmodel.add(layers.Dropout(<dropout>))"],
            '<rnn-layers>': ["<rnn-layer-f>", "<rnn-layer>\n<rnn-layer-f>", "<rnn-layer>\n<rnn-layer>\n<rnn-layer-f>"],
            '<rnn-layer-f>': ["rnn_layer = getattr(layers, 'LSTM')\nmodel.add(rnn_layer(<units>,activation='<activation>',return_sequences=False))"],
            '<rnn-layer>': ["rnn_layer = getattr(layers, 'LSTM')\nmodel.add(rnn_layer(<units>,activation='<activation>',return_sequences=True))"],
            # '<rnn-type>': ["SimpleRNN", "LSTM", "GRU"],
            '<units>': ['16','32','64','128'],
            '<activation>': ["relu", "tanh", "sigmoid"],
            '<dropout>': [str(u * 0.01) for u in range(0, 91, 10)]
        }

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
            
        self.population = [GE_Individual(seed=self.seed+i) for i in range(self.pop_size)]
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
        
        # Initialize phenotypes
        for individual in population:
            individual.setPhenotype(self.genotype_to_phenotype(individual.genotype))

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
        # Initialize phenotypes for any individuals that don't have them
        for individual in self.population:
            if not hasattr(individual, 'phenotype') or individual.phenotype is None:
                individual.setPhenotype(self.genotype_to_phenotype(individual.genotype))

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
        tournament_size = max(2, self.pop_size // 5)  # Adjust tournament size based on population
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent genotypes.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Individual: New child individual
        """
        # Create new child with shared seed for weight consistency
        child_seed = random.choice([parent1.seed, parent2.seed])
        child = GE_Individual(seed=child_seed)
        
        if random.random() < self.crossover_rate:
            # Perform actual crossover
            point = random.randint(1, min(len(parent1.genotype), len(parent2.genotype)) - 1)
            child.genotype = parent1.genotype[:point] + parent2.genotype[point:]
        else:
            # No crossover, just clone one parent
            child.genotype = random.choice([parent1.genotype, parent2.genotype]).copy()
        
        # Set phenotype based on the new genotype
        child.setPhenotype(self.genotype_to_phenotype(child.genotype))
        return child
    
    def mutate(self, individual):
        """
        Mutates an individual's genotype by changing random genes.
        
        Args:
            individual: Individual to mutate
        """
        # Make a copy of the genotype to avoid modifying the original
        genotype = individual.genotype.copy()
        
        # Apply mutation with probability based on mutation rate
        for i in range(len(genotype)):
            if random.random() < self.mutation_rate:
                genotype[i] = random.randint(0, 255)
        
        # Update individual's genotype and phenotype
        individual.genotype = genotype
        individual.setPhenotype(self.genotype_to_phenotype(genotype))

    def evolve(self):
        """
        Run the grammatical evolution process.
        
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
            if self.best_individual is not None:
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
            print("\nStopped early at generation:", self.current_generation+1)
        else:
            # Final evaluation
            self.current_generation = self.generations
            print("\nFinal evaluation")
            self.evaluate_population()
        
        print(f"\nEvolution complete!")
        # print(f"Best perplexity: {self.surrogate.best_perplexity:.4f}")
        # print(f"Best architecture: {self.best_individual}")
        self.best_individual.fitness = -self.surrogate.best_perplexity
        return self.best_individual

    def genotype_to_phenotype(self, genotype):
        """
        Converts a genotype (list of integers) into a phenotype (valid RNN architecture code).
        """
        return self.expand('<rnn-model>', genotype, 0)

    def expand(self, non_terminal, genotype, index):
        """
        Expands a non-terminal using the genotype and the grammar rules.
        
        Args:
            non_terminal: The non-terminal symbol to expand
            genotype: The genotype (list of integers)
            index: Current index in the genotype
            
        Returns:
            The expanded string
        """
        if non_terminal not in self.grammar:
            return non_terminal
        
        # Select rule based on genotype value
        rule_index = genotype[index % len(genotype)] % len(self.grammar[non_terminal])
        rule = self.grammar[non_terminal][rule_index]
        
        # Find all non-terminals in the rule
        non_terminals = []
        i = 0
        while i < len(rule):
            if rule[i] == '<':
                start = i
                # Find the closing bracket
                while i < len(rule) and rule[i] != '>':
                    i += 1
                if i < len(rule):  # Found closing bracket
                    non_terminals.append((start, i + 1, rule[start:i + 1]))
            i += 1
        
        # Replace all non-terminals with their expansions
        result = rule
        offset = 0
        for start, end, nt in non_terminals:
            # Recursively expand the non-terminal
            expansion = self.expand(nt, genotype, index + 1)
            # Replace in the result string, adjusting for previous expansions
            result = result[:start + offset] + expansion + result[end + offset:]
            # Update offset based on difference in length
            offset += len(expansion) - (end - start)
        
        return result