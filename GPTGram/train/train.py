import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from ..config import Config as cfg
from ..preprocessing import WhatsDataset

class Trainer:

    def __init__(self):
        pass

    def init_model(self):
        pass

    def init_optimizer(self):
        pass

    def init_dataloader(self):
        pass

    def batch_loss(self, batch):
        pass

    def save_model(self):
        pass

    def train(self):
        pass