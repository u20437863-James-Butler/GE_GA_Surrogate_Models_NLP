import random
import numpy as np
from tensorflow import keras
from keras import layers
import hashlib

class Individual:
    def __init__(self, seed=None, layer_counts=None, rnn_types=None, units=None, activations=None, return_sequences=None, dropout=None):
        # Set seed if provided or generate new one
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        
        # Initialize attributes with defaults or provided values
        self.layer_counts = layer_counts if layer_counts is not None else random.randint(1, 3)
        self.rnn_types = rnn_types if rnn_types is not None else [random.choice(['SimpleRNN', 'LSTM', 'GRU']) for _ in range(self.layer_counts)]
        self.units = units if units is not None else [random.randint(1, 10) * 10 for _ in range(self.layer_counts)]
        self.activations = activations if activations is not None else [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(self.layer_counts)]
        
        # Fix the return_sequences to ensure last layer is False
        if return_sequences is not None:
            self.return_sequences = return_sequences
        else:
            self.return_sequences = [random.choice([True, False]) for _ in range(self.layer_counts)]
            if self.layer_counts > 1:
                self.return_sequences[-1] = False
        
        self.dropout = dropout if dropout is not None else random.uniform(0, 0.5)
        self.fitness = None

    def getId(self):
        """Generate a unique ID based on individual's architecture"""
        # Create a string representation of key attributes
        id_string = (
            f"LC{self.layer_counts}_" +
            f"RT{'_'.join(self.rnn_types)}_" +
            f"U{'_'.join(map(str, self.units))}_" +
            f"A{'_'.join(self.activations)}_" +
            f"RS{'_'.join(str(rs) for rs in self.return_sequences)}_" +
            f"DO{self.dropout:.3f}_" +
            f"S{self.seed}"
        )
        
        # Generate a shorter hash for the ID
        hash_id = hashlib.md5(id_string.encode()).hexdigest()[:8]
        return hash_id
    
    def copy(self):
        """Creates a deep copy of the individual"""
        return Individual(
            seed=self.seed,
            layer_counts=self.layer_counts,
            rnn_types=self.rnn_types[:],
            units=self.units[:],
            activations=self.activations[:],
            return_sequences=self.return_sequences[:],
            dropout=self.dropout
        )

    def getFitness(self):
        """Returns the fitness score"""
        return self.fitness

    def build_model(self, input_shape, output_dim):
        """Creates and returns an RNN model using Keras based on the BNF grammar"""
        # Set the seeds for reproducibility
        tf_seed = self.seed % (2**31 - 1)  # TF requires a smaller range
        keras.utils.set_random_seed(tf_seed)
        
        model = keras.Sequential()
        
        # Add the input layer
        if self.layer_counts > 0:
            rnn_layer = getattr(layers, self.rnn_types[0])
            model.add(rnn_layer(
                self.units[0], 
                activation=self.activations[0], 
                return_sequences=self.return_sequences[0], 
                input_shape=input_shape
            ))
        
        # Add the hidden layers
        for i in range(1, self.layer_counts):
            rnn_layer = getattr(layers, self.rnn_types[i])
            model.add(rnn_layer(
                self.units[i], 
                activation=self.activations[i], 
                return_sequences=self.return_sequences[i]
            ))
        
        # Add dropout and output layer
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(output_dim, activation='softmax'))
        
        # Compile the model
        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy']
        )
        
        return model
    
    def __str__(self):
        """String representation of the individual"""
        return (
            f"Individual(id={self.getId()}, "
            f"layers={self.layer_counts}, "
            f"types={self.rnn_types}, "
            f"units={self.units}, "
            f"activations={self.activations}, "
            f"return_sequences={self.return_sequences}, "
            f"dropout={self.dropout:.3f}, "
            f"seed={self.seed})"
        )