# src/utils/config_loader.py

import yaml
import os
from src.utils.config import PROJECT_ROOT

def load_config():
    """
    Loads configuration from config.yaml in the project root directory.
    """
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
