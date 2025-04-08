import random
import numpy as np
from tensorflow import keras
from keras import layers
import hashlib
from optimizers.individual import Individual

class GE_Individual(Individual):
    def __init__(self, seed=None, genotype=None, id=None):
        # Set seed if provided or generate a new one
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        
        self.fitness = None
        
        # Genotype is a list of integers (codons)
        self.genotype_length = 10
        self.genotype = genotype if genotype is not None else [self.rng.randint(0, 255) for _ in range(self.genotype_length)]
        
        # Phenotype will hold the decoded architecture
        self.phenotype = None

        # Id stored as it cannot be easily constructed by the 
        self.id = id

    def setGene(self, gene):
        self.genotype = gene
        self.phenotype = None

    def setPhenotype(self, phenotype):
        self.phenotype = phenotype
    
    def getId(self):
        """Generate a unique short ID based on the individual's architecture."""
        hash_id = hashlib.md5(self.id.encode()).hexdigest()[:8]
        return hash_id

    def getIdLong(self):
        """Generate a unique long ID based on the individual's architecture."""
        return self.id

    def copy(self):
        """Creates a deep copy of the individual."""
        return GE_Individual(
            seed=self.seed,
            genotype=self.genotype[:],
            id=id
        )

    def getFitness(self):
        """Returns the fitness score."""
        return self.fitness

    def build_model(self, input_shape, output_dim):
        """Creates and returns an RNN model using Keras based on the phenotype."""
        # Set the seed for reproducibility
        tf_seed = self.seed % (2**31 - 1)  # TensorFlow seed must be in a smaller range
        keras.utils.set_random_seed(tf_seed)
        
        phenotype = self.phenotype
        model = keras.Sequential()
        
        # Add embedding layer to convert token indices to dense vectors
        embedding_dim = 50  # Adjust as needed
        
        # Input layer - shape should be (sequence_length,) for token indices
        model.add(layers.Input(shape=(input_shape[0],)))
        
        # Embedding layer converts input to shape (batch_size, sequence_length, embedding_dim)
        model.add(layers.Embedding(output_dim, embedding_dim))
        
        print(self.getIdLong())
        
        # Add the RNN layers
        for i in range(phenotype['layer_counts']):
            rnn_layer = getattr(layers, phenotype['rnn_types'][i])
            # For all but the last RNN layer, set return_sequences=True
            return_seq = True if i < phenotype['layer_counts'] - 1 else False
            model.add(rnn_layer(
                phenotype['units'][i], 
                activation=phenotype['activations'][i], 
                return_sequences=return_seq
            ))
        
        # Add dropout and output layer
        model.add(layers.Dropout(phenotype['dropout']))
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
        return self.id