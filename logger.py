import logging
import os
from logging.handlers import RotatingFileHandler
from config_manager import ConfigManager

def setup_logger(name: str = None) -> logging.Logger:
    """
    Sets up and returns a logger with RotatingFileHandler based on configuration.
    
    Args:
        name: The name of the logger. If None, returns the root logger.
        
    Returns:
        logging.Logger: The configured logger.
    """
    config_manager = ConfigManager()
    logging_config = config_manager.logging_config
    
    level_str = logging_config.get('level', "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    
    # File logging settings
    log_file = logging_config.get('file', 'logs/app.log')
    max_bytes = logging_config.get('max_bytes', 10 * 1024 * 1024) # Default 10MB
    backup_count = logging_config.get('backup_count', 5)
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler
    handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Logger setup
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        
    return logger

