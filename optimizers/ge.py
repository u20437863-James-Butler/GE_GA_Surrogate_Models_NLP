import random
import numpy as np
from keras import Sequential, layers

class GrammaticalEvolution:
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate, max_genotype_length):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_genotype_length = max_genotype_length  # Max number of production rules in genotype
        self.grammar = {
            '<rnn-model>': ["Sequential([<rnn-layers>, <dense-layer>])"],
            '<rnn-layers>': ["<rnn-layer>", "<rnn-layer>, <rnn-layers>"],
            '<rnn-layer>': ["layers.<rnn-type>(<units>, activation='<activation>', return_sequences=<return-seq>)"],
            '<rnn-type>': ["SimpleRNN", "LSTM", "GRU"],
            '<units>': [str(u) for u in range(10, 101, 10)],  # Units between 10 and 100
            '<activation>': ["relu", "tanh", "sigmoid"],
            '<return-seq>': ["True", "False"],
            '<dense-layer>': ["layers.Dense(<output-dim>, activation='softmax')"],
            '<output-dim>': [str(u) for u in range(10, 101, 10)]  # Output dim between 10 and 100
        }
        self.population = self.initialize_population()
        self.depth_limit = _

    def initialize_population(self):
        """
        Initializes the population with random genotypes (lists of integers).
        Each integer maps to a production rule in the grammar.
        """
        return [
            [random.randint(0, len(self.grammar[key]) - 1) for key in self.grammar]
            for _ in range(self.pop_size)
        ]

    def genotype_to_phenotype(self, genotype):
        """
        Converts a genotype (list of integers) into a phenotype (valid RNN architecture code).
        """
        return self.expand('<rnn-model>', genotype, 0)

    def expand(self, non_terminal, genotype, index):
        """
        Expands a non-terminal using the genotype and the grammar rules.
        """
        if non_terminal not in self.grammar:
            return non_terminal
        rule_index = genotype[index % len(genotype)] % len(self.grammar[non_terminal])
        rule = self.grammar[non_terminal][rule_index]
        production = []
        for symbol in rule.split():
            if symbol.startswith('<'):  # Non-terminal
                production.append(self.expand(symbol, genotype, index + 1))
            else:  # Terminal
                production.append(symbol)
        return ' '.join(production)

    def mutate(self, genotype):
        """
        Mutates a genotype by changing one random aspect.
        """
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(genotype) - 1)
            genotype[idx] = random.randint(0, 255)

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parent genotypes.
        """
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            return parent1[:point] + parent2[point:]
        return parent1

    def evaluate_population(self, x_train, y_train, x_val, y_val, epochs=5):
        """
        Evaluates the entire population by generating models, training them, and computing fitness.
        """
        fitness_scores = []
        for genotype in self.population:
            model_code = self.genotype_to_phenotype(genotype)
            model = self.build_model_from_code(model_code)
            if model is None:
                fitness_scores.append(float('inf'))
                continue
            model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_val, y_val))
            _, accuracy = model.evaluate(x_val, y_val, verbose=0)
            fitness_scores.append(1 - accuracy)  # Minimizing error
        return fitness_scores

    def build_model_from_code(self, model_code):
        """
        Executes the generated model code and returns a compiled Keras model.
        """
        try:
            local_vars = {}
            exec(f"from tensorflow.keras import Sequential, layers\nmodel = {model_code}", {}, local_vars)
            model = local_vars.get("model")
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return model
        except Exception:
            return None

    def select_parents(self, fitness_scores):
        """
        Selects parents using tournament selection.
        """
        candidates = random.sample(range(self.pop_size), 2)
        return min(candidates, key=lambda i: fitness_scores[i])

    def evolve(self, x_train, y_train, x_val, y_val):
        """
        Runs the evolutionary process.
        """
        for generation in range(self.generations):
            fitness_scores = self.evaluate_population(x_train, y_train, x_val, y_val)
            new_population = []
            for _ in range(self.pop_size):
                p1 = self.select_parents(fitness_scores)
                p2 = self.select_parents(fitness_scores)
                child = self.crossover(self.population[p1], self.population[p2])
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        best_index = min(range(self.pop_size), key=lambda i: fitness_scores[i])
        return self.genotype_to_phenotype(self.population[best_index])
