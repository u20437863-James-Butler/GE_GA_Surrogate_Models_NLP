import random
import numpy as np
from optimizers.GA_Individual import Individual

class GeneticAlgorithm:
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate, input_shape, output_dim):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.population = [Individual() for _ in range(pop_size)]
        
    def evaluate_population(self, x_train, y_train, x_val, y_val, epochs=5):
        for individual in self.population:
            model = individual.build_model(self.input_shape, self.output_dim)
            model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_val, y_val))
            _, accuracy = model.evaluate(x_val, y_val, verbose=0)
            individual.fitness = 1 - accuracy  # Minimizing error

    def select_parents(self):
        """Tournament selection"""
        return min(random.sample(self.population, 2), key=lambda i: i.fitness)
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        child = Individual()
        child.layer_counts = random.choice([parent1.layer_counts, parent2.layer_counts])
        child.units = [random.choice(pair) for pair in zip(parent1.units, parent2.units)]
        child.activations = [random.choice(pair) for pair in zip(parent1.activations, parent2.activations)]
        child.dropout = random.choice([parent1.dropout, parent2.dropout])
        return child
    
    def mutate(self, individual):
        """Randomly mutates an individual's genes"""
        if random.random() < self.mutation_rate:
            individual.layer_counts = random.randint(1, 3)
        if random.random() < self.mutation_rate:
            individual.units = [random.randint(10, 100) for _ in range(individual.layer_counts)]
        if random.random() < self.mutation_rate:
            individual.activations = [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(individual.layer_counts)]
        if random.random() < self.mutation_rate:
            individual.dropout = random.uniform(0, 0.5)
        # Fix - change mut to follow more generic structure below
        # if random.random() < self.mutation_rate:
        #     r = random.randint(0,3)
        #     if r == 0: individual.layer_counts = random.randint(1, 3)
    
    def evolve(self, x_train, y_train, x_val, y_val):
        for gen in range(self.generations):
            self.evaluate_population(x_train, y_train, x_val, y_val)
            new_population = []
            for _ in range(self.pop_size):
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            self.population = new_population
        
        self.evaluate_population(x_train, y_train, x_val, y_val)
        best_individual = min(self.population, key=lambda ind: ind.fitness)
        return best_individual