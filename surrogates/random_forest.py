import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

class RandomForestSurrogate:
    """
    Surrogate model that uses a Random Forest to predict the performance of RNN architectures
    without fully training them, based on their architectural features.
    """
    def __init__(self, dataset, trainer=None, initial_models=50, train_epochs=5, retrain_interval=20, verbose=0):
        """
        Initialize the Random Forest surrogate model.
        
        Args:
            dataset: Dataset object containing training, validation and test data
            trainer: MinTrainSurrogate object to use for initial model evaluation
            initial_models: Number of models to fully evaluate to train the RF model
            train_epochs: Number of epochs to train models for initial evaluation
            retrain_interval: How often to retrain the RF model (in terms of evaluations)
            verbose: Verbosity level
        """
        self.dataset = dataset
        self.verbose = verbose
        
        # Get input shape and output dimension from dataset
        self.input_shape = dataset.get_input_shape()
        self.output_dim = dataset.get_output_dim()
        
        # Set up the trainer for initial evaluations
        if trainer is None:
            from surrogates.mintrain import MinTrainSurrogate
            self.trainer = MinTrainSurrogate(dataset, num_epochs=train_epochs, verbose=0)
        else:
            self.trainer = trainer
            
        self.initial_models = initial_models
        self.retrain_interval = retrain_interval
        
        # Initialize Random Forest model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Storage for training data for the Random Forest
        self.features = []  # Architecture features
        self.targets = []   # Actual performance values
        
        # Evaluation count
        self.eval_count = 0
        
        # Metrics to track
        self.best_perplexity = float('inf')
        self.best_individual = None
        
    def _extract_features(self, individual):
        """
        Extract numerical features from an individual's architecture.
        This needs to be implemented differently for different individual types.
        
        Args:
            individual: An Individual or CellBasedIndividual
            
        Returns:
            list: Numerical features representing the architecture
        """
        features = []
        
        # Handle standard Individual
        if hasattr(individual, 'layer_counts'):
            features.append(individual.layer_counts)
            features.append(individual.dropout)
            
            # Pad or truncate variable-length features
            max_layers = 5  # Maximum number of layers to consider
            
            # RNN types (one-hot encoded)
            rnn_types_map = {'SimpleRNN': 0, 'LSTM': 1, 'GRU': 2}
            rnn_types = [0] * max_layers
            for i in range(min(individual.layer_counts, max_layers)):
                rnn_types[i] = rnn_types_map.get(individual.rnn_types[i], 0)
            features.extend(rnn_types)
            
            # Units
            units = [0] * max_layers
            for i in range(min(individual.layer_counts, max_layers)):
                units[i] = individual.units[i] / 100.0  # Normalize
            features.extend(units)
            
            # Activations (one-hot encoded)
            act_map = {'relu': 0, 'tanh': 1, 'sigmoid': 2}
            activations = [0] * max_layers
            for i in range(min(individual.layer_counts, max_layers)):
                activations[i] = act_map.get(individual.activations[i], 0)
            features.extend(activations)
            
            # Return sequences
            return_seq = [0] * max_layers
            for i in range(min(individual.layer_counts, max_layers)):
                return_seq[i] = 1 if individual.return_sequences[i] else 0
            features.extend(return_seq)
            
        # Handle CellBasedIndividual
        elif hasattr(individual, 'num_cells'):
            features.append(individual.num_cells)
            features.append(individual.units / 100.0)  # Normalize
            features.append(individual.dropout)
            
            # Cell features
            features.append(individual.cell.num_nodes)
            
            # Operations (one-hot encoded)
            op_map = {'tanh': 0, 'sigmoid': 1, 'relu': 2, 'linear': 3, 'hard_sigmoid': 4}
            max_nodes = 5  # Maximum number of nodes to consider
            ops = [0] * max_nodes
            for i in range(min(individual.cell.num_nodes, max_nodes)):
                ops[i] = op_map.get(individual.cell.operations[i], 0)
            features.extend(ops)
            
            # Connection density
            conn_density = np.sum(individual.cell.connections) / (individual.cell.num_nodes ** 2)
            features.append(conn_density)
        
        return features
    
    def _initialize_rf_model(self, population):
        """
        Train initial models and initialize the Random Forest with their results
        
        Args:
            population: List of individuals to sample from for initial training
        """
        if self.verbose > 0:
            print(f"Initializing Random Forest model with {self.initial_models} evaluations...")
        
        # Train and evaluate initial models
        for i in range(min(self.initial_models, len(population))):
            individual = population[i]
            
            # Extract features
            features = self._extract_features(individual)
            self.features.append(features)
            
            # Train model using mintrain surrogate
            fitness = self.trainer.evaluate(individual)
            self.targets.append(fitness)
            
            # Update best individual
            if individual.fitness is not None and -individual.fitness < self.best_perplexity:
                self.best_perplexity = -individual.fitness
                self.best_individual = individual.copy()
        
        # Train the RF model
        self._train_rf_model()
        
    def _train_rf_model(self):
        """Train the Random Forest model on collected data"""
        if len(self.features) > 5:  # Need minimum samples to train
            features_array = np.array(self.features)
            targets_array = np.array(self.targets)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features_array)
            
            # Train the model
            self.rf_model.fit(scaled_features, targets_array)
            
            if self.verbose > 0:
                print(f"RF model trained on {len(self.features)} samples")
    
    def evaluate(self, individual):
        """
        Evaluate an individual using the Random Forest surrogate model
        
        Args:
            individual: An Individual or CellBasedIndividual
            
        Returns:
            float: Predicted fitness score
        """
        self.eval_count += 1
        
        # First time initialization
        if len(self.features) == 0 and self.eval_count == 1:
            # We need to evaluate at least one individual for real
            fitness = self.trainer.evaluate(individual)
            features = self._extract_features(individual)
            
            self.features.append(features)
            self.targets.append(fitness)
            
            # Update best individual
            if -fitness < self.best_perplexity:
                self.best_perplexity = -fitness
                self.best_individual = individual.copy()
                
            individual.fitness = fitness
            return fitness
        
        # Extract features from individual
        features = self._extract_features(individual)
        
        # Decide whether to use real evaluation or RF prediction
        use_real_eval = (
            len(self.features) < self.initial_models or  # Still building initial dataset
            self.eval_count % self.retrain_interval == 0  # Periodic real evaluation
        )
        
        if use_real_eval:
            # Perform actual evaluation
            fitness = self.trainer.evaluate(individual)
            
            # Store result for future RF training
            self.features.append(features)
            self.targets.append(fitness)
            
            # Update best individual
            if -fitness < self.best_perplexity:
                self.best_perplexity = -fitness
                self.best_individual = individual.copy()
                
            # Retrain the RF model periodically
            if self.eval_count % self.retrain_interval == 0:
                self._train_rf_model()
        else:
            # Use RF to predict fitness
            try:
                scaled_features = self.scaler.transform([features])
                fitness = self.rf_model.predict(scaled_features)[0]
            except:
                # Fall back to real evaluation if RF fails
                fitness = self.trainer.evaluate(individual)
                self.features.append(features)
                self.targets.append(fitness)
        
        individual.fitness = fitness
        return fitness
    
    def evaluate_population(self, population):
        """
        Evaluate a population of individuals.
        
        Args:
            population: List of Individual or CellBasedIndividual objects
            
        Returns:
            list: List of fitness scores
        """
        # Initialize RF model if needed
        if len(self.features) == 0:
            self._initialize_rf_model(population)
            
        # Evaluate remaining population
        return [self.evaluate(individual) for individual in population]
    
    def get_best_individual(self):
        """Return the best individual found so far"""
        return self.best_individual
    
    def save_model(self, filename='rf_surrogate.pkl'):
        """Save the trained RF model to disk"""
        model_data = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'features': self.features,
            'targets': self.targets
        }
        joblib.dump(model_data, filename)
        
    def load_model(self, filename='rf_surrogate.pkl'):
        """Load a trained RF model from disk"""
        model_data = joblib.load(filename)
        self.rf_model = model_data['rf_model']
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        self.targets = model_data['targets']