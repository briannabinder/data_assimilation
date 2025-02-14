# utils/logger.py
import logging
import os

def get_logger(model_name, filter_name, ensemble_size):
    """
    Set up and return a logger that logs information to a specific log file.
    
    Args:
        model_name (str): The name of the model (e.g., 'lorenz63').
        filter_name (str): The name of the filter being used (e.g., 'EnsembleKalmanFilter').
        ensemble_size (int): The ensemble size for this run.
        
    Returns:
        logger (logging.Logger): The configured logger.
    """

    # Creates a directory for the logs inside the model folder if it doesn't exist
    log_dir = f"logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f'{log_dir}{model_name}_{filter_name}_ensemble_{ensemble_size}.log'
    if os.path.exists(log_filename):
        os.remove(log_filename)  # Remove the old log file if it exists
    
    logger = logging.getLogger(f'{model_name}_{filter_name}_{ensemble_size}')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    
    return logger