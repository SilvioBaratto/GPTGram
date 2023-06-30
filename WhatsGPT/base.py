import os
import glob
import torch
from typing import Union, Tuple
import numpy as np
from abc import ABCMeta, abstractmethod
from .config import Config as cfg

class Base(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the BaseEstimator instance. It sets the attributes from the keyword arguments to the
        configuration.

        Args:
            **kwargs: Additional keyword arguments to be set as attributes in the Training configuration.

        Returns:
            None
        """
        for key, value in kwargs.items():
            setattr(cfg.gpt, key, value)
            setattr(cfg.io_metrics, key, value)
            setattr(cfg.data, key, value)
            setattr(cfg.model, key, value)
            setattr(cfg.optimizer, key, value)
            setattr(cfg.learning_rate, key, value)
            setattr(cfg.ddp, key, value)
            setattr(cfg.system, key, value)
            setattr(cfg.sampling, key, value)