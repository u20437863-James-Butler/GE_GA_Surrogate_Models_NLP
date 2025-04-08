class Individual:
    def __init__(self, seed=None, layer_counts=None, rnn_types=None, units=None, activations=None, dropout=None):
        pass

    def getId(self):
        """Generate a unique short ID based on the individual's architecture."""
        pass

    def getIdLong(self):
        """Generate a unique long ID based on the individual's architecture."""
        pass

    def copy(self):
        """Creates a deep copy of the individual."""
        pass

    def getFitness(self):
        """Returns the fitness score."""
        pass

    def build_model(self, input_shape, output_dim):
        """Creates and returns an RNN model using Keras.
        
        For all but the last RNN layer, return_sequences is set to True to ensure a 3D output.
        The last RNN layer always uses return_sequences=False.
        """
        pass
