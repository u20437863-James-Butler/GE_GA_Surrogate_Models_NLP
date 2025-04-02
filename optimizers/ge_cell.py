import random
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf

class CellBasedGrammaticalEvolution:
    def __init__(self, pop_size=50, generations=30, mutation_rate=0.1, crossover_rate=0.7, max_genotype_length=100, depth_limit=10):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_genotype_length = max_genotype_length
        self.depth_limit = depth_limit
        
        # Define the grammar for cell-based RNN architecture
        self.grammar = {
            '<cell-based-model>': ["{'num_cells': <num-cells>, 'units': <units>, 'dropout': <dropout>, 'cell': <cell>}"],
            '<num-cells>': [str(n) for n in range(1, 6)],  # 1-5 cells
            '<units>': [str(u) for u in range(10, 101, 10)],  # 10-100 units
            '<dropout>': [str(round(d/10, 1)) for d in range(0, 6)],  # 0.0-0.5 dropout
            
            # Cell structure definition
            '<cell>': ["{'num_nodes': <num-nodes>, 'operations': <operations>, 'connections': <connections>}"],
            '<num-nodes>': [str(n) for n in range(2, 6)],  # 2-5 nodes in cell
            
            # Operations for each node in the cell
            '<operations>': ["[<operation-list>]"],
            '<operation-list>': ["<operation>", "<operation>, <operation-list>"],
            '<operation>': ["'tanh'", "'sigmoid'", "'relu'", "'linear'", "'hard_sigmoid'"],
            
            # Connection matrix for the cell DAG
            '<connections>': ["self.generate_connections(<num-nodes>)"],
        }
        
        self.population = self.initialize_population()
        self.input_shape = None
        self.output_dim = None
    
    def generate_connections(self, num_nodes):
        """Generate a valid DAG connection matrix"""
        num_nodes = int(num_nodes)
        connections = np.zeros((num_nodes, num_nodes), dtype=bool)
        
        # Create upper triangular matrix (ensures DAG property)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                connections[i, j] = random.choice([True, False])
        
        # Ensure all nodes (except first) have at least one incoming connection
        for j in range(1, num_nodes):
            if not np.any(connections[:, j]):
                i = random.randint(0, j-1)
                connections[i, j] = True
        
        return connections.tolist()
    
    def initialize_population(self):
        """Initialize population with random genotypes"""
        population = []
        for _ in range(self.pop_size):
            # Create genotype with codon values in range 0-255
            genotype = [random.randint(0, 255) for _ in range(self.max_genotype_length)]
            population.append(genotype)
        return population
    
    def genotype_to_phenotype(self, genotype):
        """Convert genotype to a phenotype (dictionary representing model architecture)"""
        try:
            phenotype_str = self.expand('<cell-based-model>', genotype, 0, 0)
            
            # Convert string representation to actual Python dictionary
            local_vars = {'self': self}
            exec(f"phenotype = {phenotype_str}", {}, local_vars)
            return local_vars.get('phenotype')
        except Exception as e:
            print(f"Error in genotype to phenotype conversion: {e}")
            return None
    
    def expand(self, symbol, genotype, index, depth):
        """Recursively expand a non-terminal symbol according to grammar rules"""
        # Check depth limit to prevent infinite recursion
        if depth > self.depth_limit:
            if '<operation-list>' in symbol:
                return "'tanh'"  # Default operation if depth limit reached
            return symbol  # Return symbol as is if depth limit exceeded
        
        # Terminal symbol
        if symbol not in self.grammar:
            return symbol
        
        # Select a production rule based on genotype value
        codon = genotype[index % len(genotype)]
        rule_idx = codon % len(self.grammar[symbol])
        rule = self.grammar[symbol][rule_idx]
        
        # Recursively expand all symbols in the rule
        parts = []
        current_part = ""
        in_symbol = False
        
        idx = index + 1
        
        for char in rule:
            if char == '<':
                if current_part:
                    parts.append(current_part)
                current_part = '<'
                in_symbol = True
            elif char == '>' and in_symbol:
                current_part += '>'
                non_terminal = current_part
                expansion = self.expand(non_terminal, genotype, idx, depth + 1)
                parts.append(expansion)
                idx += 1
                current_part = ""
                in_symbol = False
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)
        
        return ''.join(parts)
    
    def build_model(self, phenotype):
        """Build Keras model from phenotype dictionary"""
        try:
            # Extract architecture parameters
            num_cells = int(phenotype['num_cells'])
            units = int(phenotype['units'])
            dropout_rate = float(phenotype['dropout'])
            cell_spec = phenotype['cell']
            
            # Build the custom RNN cell
            rnn_cell = self.build_custom_cell(cell_spec, units)
            
            # Create the model
            model = keras.Sequential()
            
            # Add RNN layers with the custom cell
            for i in range(num_cells - 1):
                model.add(keras.layers.RNN(
                    rnn_cell,
                    return_sequences=True,
                    input_shape=self.input_shape if i == 0 else None
                ))
            
            # Last RNN layer
            model.add(keras.layers.RNN(
                rnn_cell,
                return_sequences=False,
                input_shape=self.input_shape if num_cells == 1 else None
            ))
            
            # Add dropout and output layer
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(self.output_dim, activation='softmax'))
            
            # Compile model
            model.compile(loss='sparse_categorical_crossentropy', 
                         optimizer='adam', 
                         metrics=['accuracy'])
            
            return model
        except Exception as e:
            print(f"Error building model: {e}")
            return None
    
    def build_custom_cell(self, cell_spec, units):
        """Build a custom RNN cell from cell specification"""
        num_nodes = int(cell_spec['num_nodes'])
        operations = cell_spec['operations']
        connections = np.array(cell_spec['connections'])
        
        # Ensure we have enough operations
        while len(operations) < num_nodes:
            operations.append('tanh')  # Default operation
        
        # Create custom RNN cell
        class CustomRNNCell(keras.layers.Layer):
            def __init__(self, parent, num_nodes, operations, connections, units, **kwargs):
                self.parent = parent
                self.num_nodes = num_nodes
                self.operations = operations
                self.connections = connections
                self.units = units
                self.state_size = [units]
                super(CustomRNNCell, self).__init__(**kwargs)
            
            def build(self, input_shape):
                # Input projection weights
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer='glorot_uniform',
                    name='kernel'
                )
                
                # Recurrent weights
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    initializer='orthogonal',
                    name='recurrent_kernel'
                )
                
                # Node-specific weights
                self.node_kernels = []
                for i in range(self.num_nodes):
                    self.node_kernels.append(self.add_weight(
                        shape=(self.units, self.units),
                        initializer='glorot_uniform',
                        name=f'node_kernel_{i}'
                    ))
                
                self.built = True
            
            def call(self, inputs, states):
                # Previous hidden state
                h_prev = states[0]
                
                # Process input
                x = tf.matmul(inputs, self.kernel)
                
                # Process recurrent connection
                r = tf.matmul(h_prev, self.recurrent_kernel)
                
                # Combine input and recurrent
                c = x + r
                
                # Process through the cell DAG
                node_outputs = [None] * self.num_nodes
                node_outputs[0] = c  # First node is the combined input
                
                # Process each node
                for i in range(1, self.num_nodes):
                    # Collect inputs from connected nodes
                    node_inputs = []
                    for j in range(i):
                        if self.connections[j, i]:
                            node_inputs.append(node_outputs[j])
                    
                    if node_inputs:
                        # Combine inputs if multiple
                        if len(node_inputs) > 1:
                            combined = tf.add_n(node_inputs)
                        else:
                            combined = node_inputs[0]
                        
                        # Apply node transformation
                        transformed = tf.matmul(combined, self.node_kernels[i])
                        
                        # Apply activation
                        activation_fn = getattr(tf.nn, self.operations[i])
                        node_outputs[i] = activation_fn(transformed)
                    else:
                        # Fallback (shouldn't happen with our constraints)
                        node_outputs[i] = node_outputs[0]
                
                # Final output is the last node
                output = node_outputs[-1]
                
                return output, [output]
        
        return CustomRNNCell(self, num_nodes, operations, connections, units)
    
    def mutate(self, genotype):
        """Mutate the genotype by randomly changing some codons"""
        for i in range(len(genotype)):
            if random.random() < self.mutation_rate:
                genotype[i] = random.randint(0, 255)
        return genotype
    
    def crossover(self, parent1, parent2):
        """Perform one-point crossover between two parents"""
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child = parent1[:point] + parent2[point:]
            return child
        return parent1.copy()
    
    def select_parents(self, fitness_scores):
        """Tournament selection to pick parents"""
        idx1 = random.randint(0, len(fitness_scores) - 1)
        idx2 = random.randint(0, len(fitness_scores) - 1)
        if fitness_scores[idx1] < fitness_scores[idx2]:
            return idx1
        return idx2
    
    def evaluate_population(self, x_train, y_train, x_val, y_val, epochs=3):
        """Evaluate all individuals in the population"""
        fitness_scores = []
        
        for genotype in self.population:
            phenotype = self.genotype_to_phenotype(genotype)
            
            if phenotype is None:
                fitness_scores.append(float('inf'))
                continue
            
            model = self.build_model(phenotype)
            
            if model is None:
                fitness_scores.append(float('inf'))
                continue
            
            try:
                # Use early stopping to prevent wasting time on poor models
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=2, restore_best_weights=True
                )
                
                # Train the model
                model.fit(
                    x_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    verbose=0,
                    callbacks=[early_stopping]
                )
                
                # Evaluate the model
                _, accuracy = model.evaluate(x_val, y_val, verbose=0)
                fitness = 1 - accuracy  # Minimize error
                
                # Add complexity penalty (optional)
                # This encourages finding simpler models when performance is similar
                num_cells = int(phenotype['num_cells'])
                num_nodes = int(phenotype['cell']['num_nodes'])
                complexity_penalty = 0.001 * (num_cells * num_nodes)
                
                fitness_scores.append(fitness + complexity_penalty)
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                fitness_scores.append(float('inf'))
                
            # Clear Keras session to avoid memory leaks
            tf.keras.backend.clear_session()
                
        return fitness_scores
    
    def evolve(self, x_train, y_train, x_val, y_val, input_shape, output_dim):
        """Run the evolution process"""
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        best_genotype = None
        best_fitness = float('inf')
        best_phenotype = None
        
        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")
            
            # Evaluate current population
            fitness_scores = self.evaluate_population(x_train, y_train, x_val, y_val)
            
            # Find best individual
            min_idx = fitness_scores.index(min(fitness_scores))
            current_best_fitness = fitness_scores[min_idx]
            
            # Update best overall
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_genotype = self.population[min_idx].copy()
                best_phenotype = self.genotype_to_phenotype(best_genotype)
                print(f"New best fitness: {1-best_fitness:.4f}, phenotype: {best_phenotype}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individual
            new_population.append(self.population[min_idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.pop_size:
                # Select parents
                p1_idx = self.select_parents(fitness_scores)
                p2_idx = self.select_parents(fitness_scores)
                
                # Create child through crossover and mutation
                child = self.crossover(self.population[p1_idx], self.population[p2_idx])
                child = self.mutate(child)
                
                new_population.append(child)
            
            # Replace population
            self.population = new_population
        
        # Build and return best model found
        best_model = None
        if best_phenotype:
            best_model = self.build_model(best_phenotype)
            
        return best_model, best_phenotype
    
    def describe_cell(self, phenotype):
        """Generate a human-readable description of the cell architecture"""
        if not phenotype:
            return "Invalid phenotype"
        
        num_cells = int(phenotype['num_cells'])
        units = int(phenotype['units'])
        dropout = float(phenotype['dropout'])
        cell = phenotype['cell']
        num_nodes = int(cell['num_nodes'])
        operations = cell['operations']
        connections = np.array(cell['connections'])
        
        description = []
        description.append(f"Cell-based RNN with {num_cells} stacked cell(s), {units} units, and {dropout} dropout")
        description.append(f"Cell structure: {num_nodes} nodes with operations: {operations}")
        description.append("Connections:")
        
        for i in range(num_nodes):
            outgoing = []
            for j in range(num_nodes):
                if connections[i, j]:
                    outgoing.append(j)
            description.append(f"  Node {i} -> {outgoing}")
        
        return "\n".join(description)