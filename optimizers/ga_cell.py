import random
import numpy as np
from optimizers.GA_cell_Individual import CellBasedIndividual

class CellBasedGeneticAlgorithm:
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate, input_shape, output_dim):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.population = [CellBasedIndividual() for _ in range(pop_size)]
        
    def evaluate_population(self, x_train, y_train, x_val, y_val, epochs=5):
        for individual in self.population:
            try:
                model = individual.build_model(self.input_shape, self.output_dim)
                model.fit(x_train, y_train, epochs=epochs, verbose=0, validation_data=(x_val, y_val))
                _, accuracy = model.evaluate(x_val, y_val, verbose=0)
                individual.fitness = 1 - accuracy  # Minimizing error
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                individual.fitness = 1.0  # Worst fitness for failed models
    
    def select_parents(self):
        """Tournament selection"""
        return min(random.sample(self.population, 2), key=lambda i: i.fitness)
    
    def crossover(self, parent1, parent2):
        """Crossover cells and architecture parameters"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        # Create new child
        child = CellBasedIndividual()
        
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
        """Mutate the individual"""
        # Mutate architectural parameters
        if random.random() < self.mutation_rate:
            individual.num_cells = random.randint(1, individual.caps["num_cells"])
        if random.random() < self.mutation_rate:
            individual.units = random.randint(10, individual.caps["units"])
        if random.random() < self.mutation_rate:
            individual.dropout = random.uniform(0, individual.caps["dropout"])
        
        # Mutate cell structure
        individual.cell.mutate(self.mutation_rate)
    
    def evolve(self, x_train, y_train, x_val, y_val):
        best_fitness = float('inf')
        best_individual = None
        
        for gen in range(self.generations):
            self.evaluate_population(x_train, y_train, x_val, y_val)
            
            # Track best individual
            current_best = min(self.population, key=lambda ind: ind.fitness)
            if current_best.fitness < best_fitness:
                best_fitness = current_best.fitness
                best_individual = current_best.copy()
                print(f"Generation {gen}: New best fitness: {1 - best_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep the best individual
            new_population.append(current_best.copy())
            
            # Generate rest of population through selection, crossover, mutation
            while len(new_population) < self.pop_size:
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
        
        # Final evaluation to ensure best individual has an accurate fitness
        self.evaluate_population(x_train, y_train, x_val, y_val)
        final_best = min(self.population, key=lambda ind: ind.fitness)
        
        # Return the best individual found across all generations
        if final_best.fitness < best_fitness:
            return final_best
        else:
            return best_individual