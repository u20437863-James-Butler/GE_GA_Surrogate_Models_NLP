import random 
import numpy as np
from tensorflow import keras
from keras import layers
import hashlib
import re
from optimizers.individual import Individual

class GE_Individual(Individual):
    def __init__(self, seed=None, genotype=None, id=None):
        # Set seed if provided or generate a new one
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        
        self.fitness = None
        
        # Genotype is a list of integers (codons)
        self.genotype_length = 10
        self.genotype = genotype if genotype is not None else [random.randint(0, 255) for _ in range(self.genotype_length)]
        
        # Phenotype will hold the decoded architecture
        self.phenotype = None
        
        # Architecture details to be parsed from phenotype
        self.architecture = {
            "layer_counts": 0,
            "units": [],
            "activations": [],
            "dropout": 0.0
        }
        
        # Id stored as it cannot be easily constructed by the 
        self.id = id if id is not None else str(self.genotype)

    def setGene(self, gene):
        self.genotype = gene
        self.phenotype = None
        # Reset architecture details when genotype changes
        self.architecture = {
            "layer_counts": 0,
            "units": [],
            "activations": [],
            "dropout": 0.0
        }

    def setPhenotype(self, phenotype):
        self.phenotype = phenotype
        # Parse phenotype to extract architecture details
        self._parse_phenotype()
    
    def _parse_phenotype(self):
        """Parse phenotype to extract architecture details"""
        if not self.phenotype:
            return
        
        # Extract layer count
        rnn_layers = self.phenotype.count('rnn_layer = getattr')
        self.architecture["layer_counts"] = rnn_layers
        
        # Extract units
        units_pattern = r"model\.add\(rnn_layer\((\d+),"
        self.architecture["units"] = [int(match) for match in re.findall(units_pattern, self.phenotype)]
        
        # Extract activations
        activation_pattern = r"activation='(\w+)'"
        self.architecture["activations"] = re.findall(activation_pattern, self.phenotype)
        
        # Extract dropout
        dropout_pattern = r"Dropout\((0\.\d+)\)"
        dropout_matches = re.findall(dropout_pattern, self.phenotype)
        if dropout_matches:
            self.architecture["dropout"] = float(dropout_matches[0])

    def getId(self):
        """Generate a unique short ID based on the individual's architecture."""
        hash_id = hashlib.md5(self.id.encode()).hexdigest()[:8]
        return hash_id

    def getIdLong(self):
        """Generate a unique long ID based on the individual's architecture."""
        return self.id

    def copy(self):
        """Creates a deep copy of the individual."""
        new_individual = GE_Individual(
            seed=self.seed,
            genotype=self.genotype[:],
            id=self.id
        )
        if self.phenotype:
            new_individual.setPhenotype(self.phenotype)
        return new_individual

    def getFitness(self):
        """Returns the fitness score."""
        return self.fitness

    def build_model(self, input_shape, output_dim):
        """Creates and returns an RNN model using Keras based on the phenotype."""
        # Set the seed for reproducibility
        tf_seed = self.seed % (2**31 - 1)  # TensorFlow seed must be in a smaller range
        keras.utils.set_random_seed(tf_seed)
        
        if self.phenotype is None:
            return None
        
        try:
            # Initialize the model
            model = keras.Sequential()
            
            # Add embedding layer to convert token indices to dense vectors
            embedding_dim = 50  # Adjust as needed
            
            # Input layer - shape should be (sequence_length,) for token indices
            model.add(layers.Input(shape=(input_shape[0],)))
            
            # Embedding layer converts input to shape (batch_size, sequence_length, embedding_dim)
            model.add(layers.Embedding(output_dim, embedding_dim))
            
            print(self.getIdLong())
            
            # Execute the phenotype code to add RNN layers and dropout
            # The phenotype code expects 'model' and 'layers' to exist
            exec(self.phenotype, {"layers": layers}, {"model": model})
            
            # Add output layer
            model.add(layers.Dense(output_dim, activation='softmax'))
            
            # Compile the model
            model.compile(
                loss='sparse_categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            print(f"Error building model: {e}")
            return None

    def __str__(self):
        """String representation of the individual."""
        # Format similar to GA_Individual's __str__ method
        return (
            f"Individual(id={self.getId()}, "
            f"layers={self.architecture['layer_counts']}, "
            f"units={self.architecture['units']}, "
            f"activations={self.architecture['activations']}, "
            f"dropout={self.architecture['dropout']:.3f}, "
            f"seed={self.seed})"
        )