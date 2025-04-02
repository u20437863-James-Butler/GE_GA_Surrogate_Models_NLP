import random
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf

class Cell:
    """Represents a recurrent cell in the architecture"""
    def __init__(self, num_nodes=None, operations=None, connections=None):
        # Cell configuration limits
        self.max_nodes = 5
        self.available_ops = ['tanh', 'sigmoid', 'relu', 'linear', 'hard_sigmoid']
        
        # Initialize cell structure
        self.num_nodes = num_nodes if num_nodes is not None else random.randint(2, self.max_nodes)
        
        # Initialize operations for each node
        self.operations = operations if operations is not None else [
            random.choice(self.available_ops) for _ in range(self.num_nodes)
        ]
        
        # Initialize connection matrix (adjacency matrix defining the cell topology)
        # connections[i][j] = True means node i connects to node j
        if connections is not None:
            self.connections = connections
        else:
            # Create random DAG structure (upper triangular to ensure no cycles)
            self.connections = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    self.connections[i, j] = random.choice([True, False])
            
            # Ensure all nodes have at least one incoming connection (except node 0)
            for j in range(1, self.num_nodes):
                if not np.any(self.connections[:, j]):
                    # Connect a random previous node to this one
                    i = random.randint(0, j-1)
                    self.connections[i, j] = True
    
    def copy(self):
        """Deep copy of the cell"""
        return Cell(
            num_nodes=self.num_nodes,
            operations=self.operations[:],
            connections=self.connections.copy()
        )
    
    def mutate(self, mutation_rate=0.2):
        """Mutate the cell structure"""
        # Mutate number of nodes
        if random.random() < mutation_rate:
            old_num_nodes = self.num_nodes
            self.num_nodes = random.randint(2, self.max_nodes)
            
            # Handle connections and operations for changed node count
            if self.num_nodes > old_num_nodes:
                # Add new operations
                for _ in range(old_num_nodes, self.num_nodes):
                    self.operations.append(random.choice(self.available_ops))
                
                # Expand connection matrix
                new_connections = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
                new_connections[:old_num_nodes, :old_num_nodes] = self.connections
                
                # Add connections for new nodes
                for i in range(old_num_nodes):
                    for j in range(old_num_nodes, self.num_nodes):
                        new_connections[i, j] = random.choice([True, False])
                
                # Ensure new nodes have incoming connections
                for j in range(old_num_nodes, self.num_nodes):
                    if not np.any(new_connections[:, j]):
                        i = random.randint(0, j-1)
                        new_connections[i, j] = True
                
                self.connections = new_connections
            
            elif self.num_nodes < old_num_nodes:
                # Truncate operations and connections
                self.operations = self.operations[:self.num_nodes]
                self.connections = self.connections[:self.num_nodes, :self.num_nodes]
        
        # Mutate operations
        for i in range(self.num_nodes):
            if random.random() < mutation_rate:
                self.operations[i] = random.choice(self.available_ops)
        
        # Mutate connections
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):  # Only upper triangular to maintain DAG
                if random.random() < mutation_rate:
                    self.connections[i, j] = not self.connections[i, j]
        
        # Ensure all nodes have at least one incoming connection (except node 0)
        for j in range(1, self.num_nodes):
            if not np.any(self.connections[:, j]):
                i = random.randint(0, j-1)
                self.connections[i, j] = True


class CellBasedIndividual:
    def __init__(self, num_cells=None, units=None, dropout=None, cell=None):
        self.caps = {
            "num_cells": 5,
            "units": 100,
            "dropout": 0.5
        }
        
        # Initialize architecture parameters
        self.num_cells = num_cells if num_cells is not None else random.randint(1, self.caps["num_cells"])
        self.units = units if units is not None else random.randint(10, self.caps["units"])
        self.dropout = dropout if dropout is not None else random.uniform(0, self.caps["dropout"])
        
        # Initialize cell structure
        self.cell = cell if cell is not None else Cell()
        
        self.fitness = None
    
    def copy(self):
        """Creates a deep copy of the individual"""
        return CellBasedIndividual(
            num_cells=self.num_cells,
            units=self.units,
            dropout=self.dropout,
            cell=self.cell.copy()
        )
    
    def getFitness(self):
        """Returns the fitness score"""
        return self.fitness
    
    def build_cell(self, input_tensor, hidden_state):
        """Build the custom cell computation graph"""
        # Node outputs storage
        node_outputs = [None] * self.cell.num_nodes
        
        # Set first node as the input
        node_outputs[0] = input_tensor
        
        # Process each node in the cell
        for i in range(1, self.cell.num_nodes):
            # Collect inputs from connected nodes
            node_inputs = []
            for j in range(i):
                if self.cell.connections[j, i]:
                    node_inputs.append(node_outputs[j])
            
            if node_inputs:
                # Combine inputs if multiple connections exist
                if len(node_inputs) > 1:
                    combined = layers.Add()(node_inputs)
                else:
                    combined = node_inputs[0]
                
                # Apply operation
                activation = self.cell.operations[i]
                processed = layers.Dense(self.units, activation=activation)(combined)
                node_outputs[i] = processed
            else:
                # Fallback if somehow no connections (shouldn't happen with our constraints)
                node_outputs[i] = node_outputs[0]
        
        # Return the output of the last node
        return node_outputs[-1]
    
    def build_rnn_cell(self):
        """Creates a custom RNN cell class"""
        class CustomRNNCell(keras.layers.Layer):
            def __init__(self, parent, units, **kwargs):
                self.parent = parent
                self.units = units
                self.state_size = [units]
                super(CustomRNNCell, self).__init__(**kwargs)
            
            def build(self, input_shape):
                # Create weights for the cell
                self.kernel = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer='glorot_uniform',
                    name='kernel'
                )
                self.recurrent_kernel = self.add_weight(
                    shape=(self.units, self.units),
                    initializer='orthogonal',
                    name='recurrent_kernel'
                )
                self.built = True
            
            def call(self, inputs, states):
                prev_output = states[0]
                
                # Process input
                x = tf.matmul(inputs, self.kernel)
                
                # Process recurrent connection
                h = tf.matmul(prev_output, self.recurrent_kernel)
                
                # Combine
                combined = x + h
                
                # Use the cell structure to process the combined input
                output = self.parent.build_cell(combined, prev_output)
                
                return output, [output]
        
        return CustomRNNCell(self, self.units)
    
    def build_model(self, input_shape, output_dim):
        """Creates and returns an RNN model using the evolved cell architecture"""
        model = keras.Sequential()
        
        # Create RNN layer with custom cell
        rnn_cell = self.build_rnn_cell()
        for i in range(self.num_cells - 1):
            model.add(keras.layers.RNN(
                rnn_cell, 
                return_sequences=True,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Last RNN layer doesn't return sequences
        model.add(keras.layers.RNN(
            rnn_cell,
            return_sequences=False,
            input_shape=input_shape if self.num_cells == 1 else None
        ))
        
        # Add dropout and output layer
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(output_dim, activation='softmax'))
        
        # Compile model
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model