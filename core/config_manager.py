import json
import os
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None
    
    def __new__(cls, config_path: str = "configs/config.json"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config_path = config_path
            cls._instance.config = {}
            cls._instance.load_config()
        return cls._instance

    def load_config(self) -> None:
        """Load configuration from JSON file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                self.validate_config()
                logger.info(f"Loaded configuration from {os.path.abspath(self.config_path)}")
            except Exception as e:
                logger.error(f"Failed to load config from {os.path.abspath(self.config_path)}: {e}")
                # If load fails, keep previous config or empty? 
                # For now, if it fails, we might want to keep old or empty. 
                # If it's first load, it stays empty.
                if not self.config:
                    self.config = {}
        else:
            logger.warning(f"Config file {os.path.abspath(self.config_path)} not found! Using defaults.")
            self.config = {}

    def validate_config(self) -> None:
        """Validates the loaded configuration for required keys and types."""
        required_structure = {
            "network": ["source_type"],
            "processing": ["enable_yolo", "fps"],
            "logging": ["level"]
        }

        for section, keys in required_structure.items():
            if section not in self.config:
                logger.warning(f"Missing config section: {section}")
                self.config[section] = {}
            
            for key in keys:
                if key not in self.config[section]:
                    logger.warning(f"Missing required config key: {section}.{key}")

        # Type checks
        fps = self.get("processing", "fps")
        if fps is not None and not isinstance(fps, (int, float)):
             logger.error(f"Invalid type for processing.fps: {type(fps)}. Expected int or float.")

    def start_monitoring(self, interval: int = 2) -> None:
        """Start a background thread to monitor config file changes"""
        def monitor():
            last_mtime = 0
            if os.path.exists(self.config_path):
                last_mtime = os.path.getmtime(self.config_path)

            while True:
                try:
                    if os.path.exists(self.config_path):
                        current_mtime = os.path.getmtime(self.config_path)
                        if current_mtime > last_mtime:
                            logger.info("Config file changed, reloading...")
                            self.load_config()
                            last_mtime = current_mtime
                except Exception as e:
                    logger.error(f"Error monitoring config file: {e}")
                time.sleep(interval)
        
        t = threading.Thread(target=monitor, daemon=True)
        t.start()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Safe getter for nested config keys"""
        return self.config.get(section, {}).get(key, default)

    @property
    def network(self) -> Dict[str, Any]:
        return self.config.get('network', {})

    @property
    def processing(self) -> Dict[str, Any]:
        return self.config.get('processing', {})

    @property
    def alarm(self) -> Dict[str, Any]:
        return self.config.get('alarm', {})

    @property
    def ui(self) -> Dict[str, Any]:
        return self.config.get('ui', {})
        
    @property
    def logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})
