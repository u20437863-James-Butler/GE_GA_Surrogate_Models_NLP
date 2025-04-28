class Surrogate:
    def __init__(self, configuration=None):
        """
        Create the surrogate that will evaluate a given population
        
        Args:
            configuration: Dictionary with parameters for setting up the surrogate model
        """
        pass
    
    def evaluate_population(self, population, seed=None):
        """
        Evaluate a population of individuals.
        
        Args:
            population: List of Individual or CellBasedIndividual objects
            seed: Optional seed for weight initialization
            
        Returns:
            list: List of fitness scores
        """
        pass