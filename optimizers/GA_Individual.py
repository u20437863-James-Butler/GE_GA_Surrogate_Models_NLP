import random
import numpy as np
from tensorflow import keras
from keras import layers

class Individual:
    def __init__(self, layer_counts=None, rnn_types=None, units=None, activations=None, return_sequences=None, dropout=None):
        self.caps = {
            "layer_counts"      : 3,
            "rnn_types"         : 3,
            "units"             : 100,
            "activations"       : 3,
            "return_sequences"  : 2,
            "dropout"           : 100,
        }
        # Fix - make initialisations based on caps and grammar
        self.layer_counts = layer_counts if layer_counts is not None else random.randint(1, self.caps["layer_counts"])
        self.rnn_types = rnn_types if rnn_types is not None else [random.choice(['SimpleRNN', 'LSTM', 'GRU']) for _ in range(self.layer_counts)]
        self.units = units if units is not None else [random.randint(10, self.caps["units"]) for _ in range(self.layer_counts)]
        self.activations = activations if activations is not None else [random.choice(['relu', 'tanh', 'sigmoid']) for _ in range(self.layer_counts)]
        self.return_sequences = return_sequences if return_sequences is not None else [random.choice([True, False]) for _ in range(self.layer_counts - 1)] + [False]
        self.dropout = dropout if dropout is not None else random.uniform(0, 0.5)

        self.fitness = None
    
    def copy(self):
        """Creates a deep copy of the individual"""
        return Individual(
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
        model = keras.Sequential()
        for i in range(self.layer_counts):
            rnn_layer = getattr(layers, self.rnn_types[i])
            model.add(rnn_layer(self.units[i], activation=self.activations[i], return_sequences=self.return_sequences[i]))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(output_dim, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
