"""
H-ViT - A Hierarchical Vision Transformer for Deformable Image Registration

This package implements the H-ViT architecture for deformable medical image registration.
It provides tools for training, evaluating, and using the H-ViT model on various datasets.

Key components:
- Model definitions (HViT, HViT_Light)
- Loss functions (DiceLoss, Grad3D, DiceScore)
- Dataset loaders (OASIS_Dataset)
- Training utilities (LiTHViT trainer)
"""

import os
import logging
from datetime import datetime
from typing import Optional, Union, Dict, List
from .utils import Logger, read_yaml_file, count_parameters, get_one_hot
from .model.hvit import HViT
from .model.hvit_light import HViT_Light
from .model.transformation import SpatialTransformer
from .data.datasets import OASIS_Dataset, get_dataloader
from .loss import DiceLoss, Grad3D, DiceScore, loss_functions
from .trainer import LiTHViT
from .scripts.main import main as run_main

# Package version
__version__ = "0.1.0"

# Set up default paths
checkpoints_dir = "./checkpoints"
logs_dir = "./logs"

# Define a lazy logger that's only created when first accessed
_logger = None

def get_logger():
    """
    Get or initialize the package logger.
    
    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger

# Define functions for setting up logging and checkpoints when needed
def setup_logging(log_dir: Optional[str] = None) -> 'Logger':
    """
    Set up logging for the application.
    
    Args:
        log_dir: Custom log directory. If None, a timestamped directory will be created.
        
    Returns:
        Logger instance
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir or f"{logs_dir}/{current_time}"
    os.makedirs(log_path, exist_ok=True)
    return Logger(log_path)

def setup_checkpoints(checkpoint_dir: Optional[str] = None) -> str:
    """
    Set up checkpoint directory for model saving.
    
    Args:
        checkpoint_dir: Custom checkpoint directory. If None, a timestamped directory will be created.
        
    Returns:
        Path to the checkpoint directory
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = checkpoint_dir or f"{checkpoints_dir}/{current_time}"
    os.makedirs(checkpoint_path, exist_ok=True)
    return checkpoint_path

def configure(config: Optional[Dict] = None) -> None:
    """
    Configure the hvit package with custom settings.
    
    Args:
        config: Configuration dictionary with optional keys:
               - 'checkpoints_dir': Directory for saving model checkpoints
               - 'logs_dir': Directory for saving logs
    """
    global checkpoints_dir, logs_dir
    
    if config is None:
        return
        
    if 'checkpoints_dir' in config:
        checkpoints_dir = config['checkpoints_dir']
        
    if 'logs_dir' in config:
        logs_dir = config['logs_dir']

# Define what gets imported with "from hvit import *"
__all__ = [
    # Models
    "HViT", "HViT_Light", "SpatialTransformer",
    
    # Data
    "OASIS_Dataset", "get_dataloader",
    
    # Loss and metrics
    "DiceLoss", "Grad3D", "DiceScore", "loss_functions",
    
    # Training
    "LiTHViT", "run_main",
    
    # Utilities
    "Logger", "read_yaml_file", "count_parameters", "get_one_hot",
    "setup_logging", "setup_checkpoints", "get_logger", "configure",
    
    # Constants and package info
    "checkpoints_dir", "logs_dir", "__version__"
]

