import os
import json

class ConfigLoader:
    """
    A class for loading and parsing configuration files for neural architecture search experiments.
    """
    
    def __init__(self, config_dir=None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_dir (str, optional): Directory containing configuration files.
                                       If None, defaults to 'configs' directory.
        """
        if config_dir is None:
            # Try to find the configs directory relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_dir = current_dir
        else:
            self.config_dir = config_dir
    
    def load_config(self, config_filename='default.json'):
        """
        Load a configuration file.
        
        Args:
            config_filename (str): Name of the configuration file to load.
                                  Defaults to 'default.json'.
        
        Returns:
            dict: Configuration parameters as a dictionary.
        
        Raises:
            FileNotFoundError: If the configuration file cannot be found.
            json.JSONDecodeError: If the configuration file is not valid JSON.
        """
        config_path = os.path.join(self.config_dir, config_filename)
        
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                print(f"Successfully loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            print(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            print(f"Invalid JSON in configuration file: {config_path}")
            raise