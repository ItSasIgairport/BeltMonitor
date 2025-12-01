import json
import os
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None
    
    def __new__(cls, config_path="config.json"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config_path = config_path
            cls._instance.config = {}
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        """Load configuration from JSON file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self.config = {}
        else:
            logger.warning(f"Config file {self.config_path} not found! Using defaults.")
            self.config = {}

    def get(self, section, key, default=None):
        """Safe getter for nested config keys"""
        return self.config.get(section, {}).get(key, default)

    @property
    def network(self):
        return self.config.get('network', {})

    @property
    def processing(self):
        return self.config.get('processing', {})

    @property
    def alarm(self):
        return self.config.get('alarm', {})

    @property
    def ui(self):
        return self.config.get('ui', {})
        
    @property
    def logging_config(self):
        return self.config.get('logging', {})

