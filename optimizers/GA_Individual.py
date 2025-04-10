import random
import numpy as np
from tensorflow import keras
from keras import layers
import hashlib
from optimizers.individual import Individual

class GA_Individual(Individual):
    def __init__(self, seed=None, layer_counts=None, rnn_types=None, units=None, activations=None, dropout=None):
        # Set seed if provided or generate a new one
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        self.fitness = None
        
        # Initialize attributes with defaults or provided values
        self.layer_counts = layer_counts if layer_counts is not None else random.randint(1, 3)
        # self.rnn_types = rnn_types if rnn_types is not None else [random.choice(['SimpleRNN', 'LSTM', 'GRU']) for _ in range(self.layer_counts)]
        self.units = units if units is not None else [random.choice([16,32,64,128]) for _ in range(self.layer_counts)]
        self.activations = activations if activations is not None else [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(self.layer_counts)]
        self.dropout = dropout if dropout is not None else random.randint(0,9) * 0.1

    def getId(self):
        """Generate a unique short ID based on the individual's architecture."""
        id_string = (
            f"LC{self.layer_counts}_" +
            # f"RT{'_'.join(self.rnn_types)}_" +
            f"U{'_'.join(map(str, self.units))}_" +
            f"A{'_'.join(self.activations)}_" +
            f"DO{self.dropout:.3f}_" +
            f"S{self.seed}"
        )
        hash_id = hashlib.md5(id_string.encode()).hexdigest()[:8]
        return hash_id

    def getIdLong(self):
        """Generate a unique long ID based on the individual's architecture."""
        id_string = (
            f"LC{self.layer_counts}_" +
            # f"RT{'_'.join(self.rnn_types)}_" +
            f"U{'_'.join(map(str, self.units))}_" +
            f"A{'_'.join(self.activations)}_" +
            f"DO{self.dropout:.3f}_" +
            f"S{self.seed}"
        )
        return id_string

    def copy(self):
        """Creates a deep copy of the individual."""
        return GA_Individual(
            seed=self.seed,
            layer_counts=self.layer_counts,
            # rnn_types=self.rnn_types[:],
            units=self.units[:],
            activations=self.activations[:],
            dropout=self.dropout
        )

    def getFitness(self):
        """Returns the fitness score."""
        return self.fitness

    def build_model(self, input_shape, output_dim):
        """Creates and returns an RNN model using Keras.
        
        For all but the last RNN layer, return_sequences is set to True to ensure a 3D output.
        The last RNN layer always uses return_sequences=False.
        """
        # Set the seed for reproducibility
        tf_seed = self.seed % (2**31 - 1)  # TensorFlow seed must be in a smaller range
        keras.utils.set_random_seed(tf_seed)
        
        model = keras.Sequential()
        
        # Add embedding layer to convert token indices to dense vectors
        embedding_dim = 50  # Adjust as needed
        
        # Input layer - shape should be (sequence_length,) for token indices
        model.add(layers.Input(shape=(input_shape[0],)))
        
        # Embedding layer converts input to shape (batch_size, sequence_length, embedding_dim)
        model.add(layers.Embedding(output_dim, embedding_dim))
        
        print(self.getIdLong())
        
        # Add the RNN layers
        for i in range(self.layer_counts):
            # For all but the last RNN layer, set return_sequences=True
            return_seq = True if i < self.layer_counts - 1 else False
            model.add(layers.LSTM(
                self.units[i], 
                activation=self.activations[i], 
                return_sequences=return_seq
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
        """String representation of the individual."""
        return (
            f"Individual(id={self.getId()}, "
            f"layers={self.layer_counts}, "
            # f"types={self.rnn_types}, "
            f"units={self.units}, "
            f"activations={self.activations}, "
            f"dropout={self.dropout:.3f}, "
            f"seed={self.seed})"
        )
