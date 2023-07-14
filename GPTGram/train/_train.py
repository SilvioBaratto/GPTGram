import os
import time
import math
import inspect
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from ..config import Config as cfg
from ..preprocessing import GramDataset
from ..model import GPT
import csv

def log_to_csv(filename: str, log_data: dict):
    """
    Logs given data to a CSV file.

    Parameters:
        filename (str): The name of the file to write to.
        log_data (dict): The data to write to the file.
    """

    # If the file does not exist, create it and write the header
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as csv_file:
            fieldnames = ['iter', 'train_loss', 'val_loss', 'lr', 'mfu']
            log_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            log_writer.writeheader()

    # Write the log data to the file
    with open(filename, 'a', newline='') as csv_file:
        log_writer = csv.DictWriter(csv_file, fieldnames=log_data.keys())
        log_writer.writerow(log_data)


def _print_parameter_info(decay_params, nodecay_params):
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")


def get_lr(it):
    """
    Compute the learning rate for the current training iteration using a cosine decay with warmup schedule.

    The learning rate schedule consists of three phases:
    1) Linear warmup for a number of steps specified by `cfg.learning_rate.warmup_iters`.
    2) Cosine decay until `cfg.learning_rate.lr_decay_iters` steps.
    3) Constant learning rate equal to `cfg.learning_rate.min_lr` after `cfg.learning_rate.lr_decay_iters` steps.

    Args:
        it (int): The current training iteration.

    Returns:
        float: The learning rate for the current training iteration.

    Raises:
        AssertionError: If the decay ratio is not in the range [0, 1].

    Note:
        The learning rate, warmup steps, decay steps, and minimum learning rate are all specified in the
        configuration object `cfg.learning_rate`.
    """
    # learning rate decay scheduler (cosine with warmup)
    if it < cfg.learning_rate.warmup_iters:
        return cfg.learning_rate.learning_rate * it / cfg.learning_rate.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.learning_rate.lr_decay_iters:
        return cfg.learning_rate.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.learning_rate.warmup_iters) / (cfg.learning_rate.lr_decay_iters - cfg.learning_rate.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.learning_rate.min_lr + coeff * (cfg.learning_rate.learning_rate - cfg.learning_rate.min_lr)

class GramTrainer:
    """
    The GramTrainer class provides a high-level API for training models with gradient accumulation and model 
    specific configuration.

    Attributes:
        train_dataloader: DataLoader for the training data.
        val_dataloader: DataLoader for the validation data.

    Note:
        The train and validation data are expected to be in binary format and are loaded from files whose 
        paths are provided during the initialization of the class.
    """

    def __init__(self, 
                 filepath: str = None, 
                 **kwargs):
        """
        Initializes the trainer with the file path for the train and validation data, the model configuration,
        the model, the optimizer, and the scaler.

        Args:
            filepath: The directory path of the binary data files 'train.bin' and 'val.bin'.
            **kwargs: Additional keyword arguments for initializing the model configuration.
        """

        self._init_paths(filepath)
        self._check_files_exist()

        self.train_dataloader = self.init_dataloader(self.train_file)
        self.val_dataloader = self.init_dataloader(self.val_file)

        self._init_config(**kwargs)
        self._init_file_paths()
        self.init_model()
        self.init_optimizer()
        self._init_scaler()

        self._init_ctx()
        self._update_gradient_accumulation_steps()

    def _init_paths(self, filepath: str) -> None:
        """
        Initializes the paths for the training and validation data files.

        Args:
            filepath: The directory path of the binary data files 'train.bin' and 'val.bin'.
        """

        self.train_file = os.path.join(filepath, 'train.bin')
        self.val_file = os.path.join(filepath, 'val.bin')

    def _check_files_exist(self):
        """
        Checks if the training and validation data files exist, raising a FileNotFoundError if they do not.
        """

        for file in [self.train_file, self.val_file]:
            if not os.path.exists(file):
                raise FileNotFoundError(f'{file} not found.')

    def _init_ctx(self) -> None:
        """
        Initializes the context manager for mixed precision training.
        """

        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16,
                   'float16': torch.float16}[cfg.system.dtype]
        
        self.ctx = nullcontext() if cfg.system.use_cuda else torch.amp.autocast(device_type='cuda', 
                                                                                dtype=ptdtype)

    def _update_gradient_accumulation_steps(self) -> None:
        """
        Updates the number of gradient accumulation steps based on the DDP world size.
        """

        if cfg.ddp.ddp:
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            assert cfg.data.gradient_accumulation_steps % ddp_world_size == 0
            cfg.data.gradient_accumulation_steps //= ddp_world_size


    def _init_config(self, **kwargs) -> None:
        """
        Initialize the configuration for the trainer.

        This method first loads the default configuration from the 'config.py' file.
        It then updates the configuration with any keyword arguments passed to the method.
        Finally, it sets the seed for the random number generator to ensure reproducibility.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None

        Raises:
            None
        """
        # Update the configuration with any keyword arguments passed to the method
        for key, value in kwargs.items():
            # list all subconfigurations in a dictionary
            subconfigs = {
                "gpt": cfg.gpt,
                "io_metrics": cfg.io_metrics,
                "data": cfg.data,
                "optimizer": cfg.optimizer,
                "learning_rate": cfg.learning_rate,
                "ddp": cfg.ddp,
                "system": cfg.system,
                "sampling": cfg.sampling
            }
            for subconfig_name, subconfig in subconfigs.items():
                if hasattr(subconfig, key):
                    setattr(subconfig, key, value)
                    break
            else:  # if no break, attribute was not found in any subconfig
                raise ValueError(f"Invalid config key: {key}")

    def _init_file_paths(self) -> None:
        """
        Initializes the file paths for saving the model state and the training log.

        This method first determines the library directory based on the "cfg.io_metrics.folder" attribute.
        It then builds the file path for saving the model state and the training log.

        Args:
            None

        Returns:
            None
        """
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(cfg.io_metrics.out_dir, file_format.format(*args))
        
        # Determine the library directory based on the "cfg.io_metrics.folder" attribute
        if cfg.io_metrics.out_dir is None:
            self.lib_dir = os.path.dirname(os.path.realpath(__file__)) # Use the current directory
        else:
            self.lib_dir = cfg.io_metrics.out_dir

        # Get the file path configurations from the '_log_build_file_path' method
        file_path_configs = self._log_build_file_path()

        # Build the file path for saving the model
        self.file_path = build_file_path(file_path_configs['file_format'], *file_path_configs['args'])

    def init_model(self) -> None:
        """
        Initializes the model for training.

        This method initializes a model for training. It supports starting from a
        pretrained model or resuming from a checkpoint. 

        If the model is initialized with CUDA support, it will be moved to the 
        appropriate device. 

        Returns:
            model (GPT): The initialized model.
        """
        # Initialize a new instance of the model
        model = GPT()
        self.model_args = dict(n_layer = cfg.gpt.n_layer,
                               n_head = cfg.gpt.n_head,
                               n_embd = cfg.gpt.n_embd,
                               block_size = cfg.gpt.block_size,
                               bias = cfg.gpt.bias,
                               vocab_size = cfg.gpt.vocab_size,
                               dropout = cfg.gpt.dropout
                               )
        
        # Device setup
        self.device = int(os.environ["LOCAL_RANK"]) if cfg.ddp.ddp \
                    else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If specified in the configuration, resume training from a checkpoint
        if cfg.io_metrics.init_from == 'resume':
            print(f"Resuming from checkpoint {self.file_path}")
            self._load_model(model)

        # Alternatively, If specified in the configuration, initialize from a pretrained model with
        # pretrained GPT-2 weights
        elif cfg.io_metrics.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights {cfg.io_metrics.init_from}")
            model = GPT.from_pretrained(cfg.io_metrics.init_from)
   
        # Move the model to the appropriate device
        self.model = model.to(self.device)

        if cfg.system.compile:
            self.model = torch.compile(self.model)

        # wrap model into DDP container
        if cfg.ddp.ddp:
            self.model = DDP(self.model, device_ids=[self.device])

    def init_optimizer(self) -> None:
        """
        Configures the optimizer for the GPT model.

        This method first collects all the model parameters that require gradients.
        It then groups these parameters into two groups based on their dimensionality.
        Any parameters that are 2D will have weight decay applied to them; all others will not.
        The method then creates an AdamW optimizer with the given learning rate, betas, and weight decay settings.
        The method uses the fused version of AdamW if it is available and if the device type is CUDA.

        Args:
            weight_decay (float): The weight decay (L2 penalty) to apply to the parameters.
            learning_rate (float): The learning rate for the optimizer.
            betas (tuple): The coefficients used for computing running averages of gradient and its square.
            device_type (str): The type of device to run the model on. Can be 'cpu' or 'cuda'.

        Returns:
            torch.optim.AdamW: The configured AdamW optimizer.

        Examples:
            >>> gpt = GPT()
            >>> optimizer = gpt.configure_optimizers(0.01, 0.001, (0.9, 0.999), 'cuda')
        """
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Group the parameters based on their dimensionality
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]  # 2D parameters will have weight decay
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]  # non-2D parameters will not have weight decay

        # Define optimizer groups with different weight decay settings
        optim_groups = [
            {'params': decay_params, 'weight_decay': cfg.optimizer.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        if cfg.ddp.ddp and self.device == 0 or not cfg.ddp.ddp:
            _print_parameter_info(decay_params, nodecay_params)

        # Check if fused AdamW is available and if the device type is CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and cfg.system.use_cuda

        # Define extra arguments for the optimizer
        extra_args = dict(fused=True) if use_fused else dict()

        # Create AdamW optimizer with the given settings
        self.optimizer = torch.optim.AdamW(optim_groups, 
                                        lr=cfg.learning_rate.learning_rate, 
                                        betas=cfg.optimizer.betas, 
                                        **extra_args)

        if not cfg.ddp.ddp or self.device == '0':
            print(f"using fused AdamW: {use_fused}")

    
    def _init_scaler(self) -> None:
        """
        Initialize the scaler for mixed precision training.

        This method first checks if the device type is CUDA and if the scaler is available.
        If both conditions are met, it initializes the scaler with the default settings.

        Args:
            None

        Returns:
            torch.cuda.amp.GradScaler: The initialized scaler.

        Examples:
            >>> scaler = self._init_scaler()
        """
        # Check if the device type is CUDA and if the scaler is available
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.system.dtype == 'float16')) \
            if cfg.system.use_cuda and torch.cuda.amp is not None else nullcontext()

    def init_dataloader(self, filepath: str) -> DataLoader:
        dataset = GramDataset(filepath)  # Create a dataset instance

        dataloader = DataLoader(dataset,
                                batch_size=cfg.data.batch_size,
                                num_workers=cfg.system.num_workers,
                                pin_memory = True,
                                shuffle = False,
                                sampler = DistributedSampler(dataset) if cfg.ddp.ddp else None
                                )

        return dataloader


    def _save_model(self, 
                    iter_num: int,
                    best_val_loss: float = None)-> None:
        """
        Save the current state of the model to a checkpoint file.

        Returns:
            None
        """
        # Create the necessary directory structure for the file path
        os.makedirs(os.path.dirname(self.file_path), mode=0o755, exist_ok=True)

        # The following section is your provided code incorporated into this function:
        raw_model = self.model.module if cfg.ddp.ddp else self.model # unwrap DDP container if needed

        state = {
            'model': raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'model_args': self.model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': self.config_to_dict()
        }

        print(f"saving checkpoint to {self.lib_dir}")
        torch.save(state, self.file_path)


    def _load_model(self,
                    model, 
                    optimizer=None) -> None:
        """
        Load a previously saved model and optimizer state from a checkpoint file.

        Args:
            model: The model object to load the saved state into.
            optimizer (optional): The optimizer object to load the saved state into. 

        Returns:
            None

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        
        # Check if the file path exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File path '{self.file_path}' does not exist")
        
        # Load the state from the file path
        checkpoint = torch.load(self.file_path, 
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Update the 'model_args', 'iter_num', 'best_val_loss', and 'config' attributes
        self.model_args = checkpoint['model_args']
        self.iter_num = checkpoint['iter_num']
        self.best_val_loss = checkpoint['best_val_loss']
        self.config = checkpoint['config']

        # Load the model state from the checkpoint
        state_dict = checkpoint['model']

        # Fix the keys of the state dictionary
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # Load the model state into the model
        model.load_state_dict(state_dict)

        # Load optimizer state if optimizer is provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])


    def _train(self) -> float:
        """
        Trains the model for a given number of iterations on the training data.

        This method sets the model to training mode, initializes the total loss tracker, 
        and then iterates over all batches in the training data loader. It ensures the data 
        is on the right device and checks whether backward gradient synchronization is required.

        For each batch, it performs a forward pass, computes the loss, and then performs a backward 
        pass to compute the gradients. The total loss is then updated.

        If the number of iterations is a multiple of the gradient accumulation steps or if it's the 
        last batch, gradients are potentially clipped based on the configuration, and the model 
        parameters are updated.

        Finally, the average training loss for the epoch is computed and returned.

        Returns:
            float: The average training loss for the iteration.
        """

        # Set the model to training mode. This enables operations which are only applied during training like dropout
        self.model.train()

        # The best validation loss encountered so far
        best_val_loss = 1e9

        # Initialize the variable for tracking the total loss
        total_loss = 0.0

        # Initialize validation interval as one third of the len of the training dataloader 
        val_interval = len(self.train_dataloader) // 3

        # Iterate over all batches in the training data loader
        for i, (x_batch, y_batch) in enumerate(self.train_dataloader):
            # Check if CUDA is available and if it is, use it and pin memory for faster CPU-to-GPU transfer
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Determine and set the learning rate for this iteration
            lr = get_lr(i) if cfg.learning_rate.decay_lr else cfg.learning_rate.learning_rate

            # Update learning rate in all parameter groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            self.model.require_backward_grad_sync = (i+1) % cfg.data.gradient_accumulation_steps == 0 if cfg.ddp.ddp else True

            # Perform a forward pass through the model and compute the loss
            with self.ctx:
                logits, loss = self.model(x_batch, y_batch)
                loss = loss / cfg.data.gradient_accumulation_steps

            # Perform a backward pass through the model to compute the gradients
            self.scaler.scale(loss).backward()
            total_loss += loss.item()

            if ((i+1) % cfg.data.gradient_accumulation_steps == 0) or (i + 1 == len(self.train_dataloader)):
                # Clip the gradients if a threshold is specified in the configuration
                if cfg.optimizer.grad_clip != 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)

                # Update the model parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # Check if it's time to perform validation
            if ((i+1) % val_interval == 0) or (i + 1 == len(self.train_dataloader)):
                val_loss = self._eval()
                print(f"train loss: {total_loss / val_interval:.4f}, val loss: {val_loss:.4f}")

                if cfg.io_metrics.log:
                    log_data = {
                        'iter': i,
                        'train_loss': total_loss / val_interval,
                        'val_loss': val_loss,
                        'lr': lr
                    }

                    log_to_csv('training_log.csv', log_data)

                if val_loss < best_val_loss or cfg.io_metrics.always_save_checkpoint:
                    best_val_loss = val_loss
                    self._save_model(i, best_val_loss)   

                # update val interval only if it's not the last iteration
                if (i + 1) != len(self.train_dataloader):
                    val_interval = val_interval * 2

                self.model.train()

        # Compute the average loss over all batches
        train_loss = total_loss / len(self.train_dataloader)

        # Return the average loss for the training data
        return train_loss
    
    @torch.no_grad()  
    def _eval(self) -> float:
        """
        Evaluates the model on the validation data.

        This method iterates over the validation DataLoader, performing a forward pass and computing the loss
        for each batch of data. After completing the iteration, it computes the average validation loss and
        returns it.

        Args:
            val_dl (torch.utils.data.DataLoader): The validation data loader.

        Returns:
            avg_val_loss (float): The average validation loss.

        """
        # Initialize the variable for tracking the total loss
        total_loss = 0.0

        # Set the model to evaluation mode. This disables operations like dropout.
        self.model.eval()

        # Iterate over all batches in the validation data loader
        for i, (x_batch, y_batch) in enumerate(self.val_dataloader):
            
            # Check if CUDA is available and if it is, use it and pin memory for faster CPU-to-GPU transfer
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # Perform a forward pass through the model and compute the loss
            with self.ctx:
                logits, loss = self.model(x_batch, y_batch)

            # Add the loss for this batch to the total loss
            total_loss += loss.item()

        # Compute the average loss over all batches
        avg_val_loss = total_loss / len(self.val_dataloader)

        # Return the average loss for the validation data
        return avg_val_loss

    def train(self) -> None:
        """
        Trains the model on the training data for a specified number of iterations.

        This method sets up tracking of training loss, lifetime iterations of the process, iterations 
        in the current epoch, raw model (possibly unwrapped from DDP container), best validation loss 
        and running memory footprint utility. It then enters a loop for the maximum iterations 
        specified in the configuration.

        In each iteration, it adjusts the learning rate, trains for one epoch, and performs evaluations 
        and logs at regular intervals. If the validation loss is better than the best so far or if 
        always_save_checkpoint is enabled in the configuration, it saves the model. It also updates 
        the memory footprint utility at intervals specified in the configuration.

        Returns:
            None
        """

        # Initialize the variable for tracking the training loss
        t0 = time.time()

        # The number of iterations in the lifetime of this process
        local_iter_num = 0

        # The number of iterations in the current epoch
        iter_num = 0

        # Unwrap the DDP container (DistributedDataParallel) if needed to get the raw model
        raw_model = self.model.module if cfg.ddp.ddp else self.model

        # Initialize the running memory footprint utility (MFU) as -1.0
        running_mfu = -1.0

        for iter_num in range(cfg.optimizer.max_iters):
                              
            # Train for one epoch
            train_loss = self._train()

            local_iter_num += 1

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Log after log_interval
            if local_iter_num % cfg.io_metrics.log_interval == 0 and (not cfg.ddp.ddp or self.device == 0):
                if local_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.data.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                print(f"iter {iter_num}: loss {train_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    def config_to_dict(self) -> dict:
        """
        list all subconfigurations in a dictionary
        """
        subconfigs = {
            **cfg.gpt.__dict__,
            **cfg.io_metrics.__dict__,
            **cfg.data.__dict__,
            **cfg.optimizer.__dict__,
            **cfg.learning_rate.__dict__,
            **cfg.ddp.__dict__,
            **cfg.system.__dict__,
            **cfg.sampling.__dict__
        }
        
        return subconfigs       

    def _log_build_file_path(self) -> dict:
        """
        Builds the file path for saving the model state.

        Returns:
            file_path_configs (dict): A dictionary containing the file format and arguments to be used for building the file path.
        """

        file_path_configs = {
            "file_format": cfg.io_metrics.run_name + '_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (cfg.gpt.block_size, cfg.gpt.vocab_size, cfg.gpt.n_layer,
                        cfg.gpt.n_head, cfg.gpt.n_embd, cfg.gpt.dropout, cfg.gpt.bias)
        }

        return file_path_configs

        


