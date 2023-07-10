import os
import time
import math
import pickle
import inspect
from contextlib import nullcontext
import wandb
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
    def __init__(self, filepath: str = None, **kwargs):
        self._init_paths(filepath)
        self._check_files_exist()

        self.train_dataloader = self.init_dataloader(self.train_file)
        self.val_dataloader = self.init_dataloader(self.val_file)

        self._init_config(**kwargs)
        self.init_model()
        self.init_optimizer()
        self._init_scaler()

        self._init_wandb()
        self._init_ctx()
        self._update_gradient_accumulation_steps()

    def _init_paths(self, filepath):
        self.train_file = os.path.join(filepath, 'train.bin')
        self.val_file = os.path.join(filepath, 'val.bin')

    def _check_files_exist(self):
        for file in [self.train_file, self.val_file]:
            if not os.path.exists(file):
                raise FileNotFoundError(f'{file} not found.')

    def _init_wandb(self):
        if cfg.io_metrics.wandb_log and (not cfg.ddp.ddp or self.device == 0):
            wandb.init(project=cfg.io_metrics.wandb_project,
                    name=cfg.io_metrics.wandb_run_name)

    def _init_ctx(self):
        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16,
                   'float16': torch.float16}[cfg.system.dtype]
        
        self.ctx = nullcontext() if cfg.system.use_cuda else torch.amp.autocast(device_type='cuda', 
                                                                                dtype=ptdtype)

    def _update_gradient_accumulation_steps(self):
        if cfg.ddp.ddp:
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            assert cfg.data.gradient_accumulation_steps % ddp_world_size == 0
            cfg.data.gradient_accumulation_steps //= ddp_world_size


    def _init_config(self, **kwargs):
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

    def init_model(self):
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

        # If specified in the configuration, resume training from a checkpoint
        if cfg.io_metrics.init_from == 'resume':
            self._load_model(model)

        # Alternatively, If specified in the configuration, initialize from a pretrained model with
        # pretrained GPT-2 weights
        elif cfg.io_metrics.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights {cfg.io_metrics.init_from}")
            model = GPT.from_pretrained(cfg.io_metrics.init_from)

        # Device setup
        self.device = int(os.environ["LOCAL_RANK"]) if cfg.ddp.ddp \
                    else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
        # Move the model to the appropriate device
        self.model = model.to(self.device)

        if cfg.system.compile:
            self.model = torch.compile(self.model)

        # wrap model into DDP container
        if cfg.ddp.ddp:
            self.model = DDP(self.model, device_ids=[self.device])

    def init_optimizer(self):
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

    
    def _init_scaler(self):
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
                                pin_memory = True,
                                shuffle = False,
                                sampler = DistributedSampler(dataset) if cfg.ddp.ddp else None
                                )

        return dataloader


    def _save_model(self, 
                    iter_num: int,
                    best_val_loss: float = None):
        """
        Save the current state of the model to a checkpoint file.

        Returns:
            None
        """
        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(cfg.io_metrics.out_dir, file_format.format(*args))
        
        # Determine the library directory based on the "cfg.io_metrics.folder" attribute
        if cfg.io_metrics.out_dir is None:
            lib_dir = os.path.dirname(os.path.realpath(__file__)) # Use the current directory
        else:
            lib_dir = cfg.io_metrics.out_dir

        # Get the file path configurations from the '_log_build_file_path' method
        file_path_configs = self._log_build_file_path()

        # Build the file path for saving the model
        file_path = build_file_path(file_path_configs['file_format'], *file_path_configs['args'])

        # Create the necessary directory structure for the file path
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

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

        print(f"saving checkpoint to {lib_dir}")
        torch.save(state, file_path)


    def _load_model(self, model, optimizer=None) -> None:
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

        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(cfg.io_metrics.out_dir, file_format.format(*args))

        # Determine the library directory based on the "cfg.io_metrics.folder" attribute
        if cfg.io_metrics.out_dir is None:
            lib_dir = os.path.dirname(os.path.realpath(__file__))  
        else:
            lib_dir = cfg.io_metrics.out_dir

        # Get the file path configurations from the '_log_build_file_path' method
        file_path_configs = self._log_build_file_path()

        # Build the file path for loading the model
        file_path = build_file_path(file_path_configs['file_format'], *file_path_configs['args'])

        # Check if the file path exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path '{file_path}' does not exist")
        
        print(f"Resuming training from {lib_dir}")

        # Load the state from the file path
        checkpoint = torch.load(file_path, map_location=self.device)
        
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


    def _train(self,
               iter_num: int) -> float:
        """
        Trains the model on the training data for the specified number of epochs.

        This method will train the model for a specified number of epochs.
        During each epoch, it will iterate over the training DataLoader, performing a forward and backward pass
        for each batch of data. Gradient accumulation is used if specified in the configuration, and the 
        gradients are clipped if a threshold is set in the configuration. After each epoch, the average 
        training loss is computed and returned.

        Args:
            train_dl (torch.utils.data.DataLoader): The training data loader.

        Returns:
            avg_train_loss (float): The average training loss for the epoch.

        """

        # Set the model to training mode. This enables operations which are only applied during training like dropout
        self.model.train()

        # Iterate over each accumulation step
        for micro_step in range(cfg.data.gradient_accumulation_steps):
            # Initialize the variable for tracking the total loss
            total_loss = 0.0

            # Iterate over all batches in the training data loader
            for x_batch, y_batch in self.train_dataloader:
                # Check if CUDA is available and if it is, use it and pin memory for faster CPU-to-GPU transfer
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Perform a forward pass through the model, getting the logits and loss
                # Note: the context manager is used to enable mixed precision training
                with self.ctx:
                    logits, loss = self.model(x_batch, y_batch)

                # Scale the loss and perform a backward pass to calculate gradients
                self.scaler.scale(loss).backward()

                # Add the unscaled loss for this batch to the total loss
                total_loss += loss.item()

            # Average the total loss by the number of gradient accumulation steps
            total_loss /= cfg.data.gradient_accumulation_steps

            if cfg.ddp.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                self.model.require_backward_grad_sync = (micro_step == cfg.data.gradient_accumulation_steps - 1)

            # Clip the gradient and step the optimizer and scaler if training in fp16
            if cfg.optimizer.grad_clip != 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Zero the gradients
            self.optimizer.zero_grad(set_to_none=True)

        # Compute the average loss over all batches
        avg_train_loss = total_loss / len(self.train_dataloader)

        # Return the average loss for this epoch
        return avg_train_loss
    
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
        for x_batch, y_batch in self.val_dataloader:
            
            # Check if CUDA is available and if it is, use it and pin memory for faster CPU-to-GPU transfer
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Perform a forward pass through the model and compute the loss
            with self.ctx:
                logits, loss = self.model(x_batch, y_batch)

            # Add the loss for this batch to the total loss
            total_loss += loss.item()

        # Compute the average loss over all batches
        avg_val_loss = total_loss / len(self.val_dataloader)

        # Return the average loss for the validation data
        return avg_val_loss

    def train(self) -> float:
        """
        The main function to handle the training process for the model.

        This function implements the complete training process including parameter updates, evaluation on the validation set, 
        timing and logging, adjusting learning rate, and saving the model with the best validation loss.

        It will continue for a number of iterations defined in the configuration.

        Returns:
            float: The best validation loss encountered during training.

        Raises:
            RuntimeError: If the training loop doesn't settle after a certain number of iterations.
        """
        
        # Initialize the variable for tracking the training loss
        t0 = time.time()

        # The number of iterations in the lifetime of this process
        local_iter_num = 0

        # The number of iterations in the current epoch
        iter_num = 0

        # Unwrap the DDP container (DistributedDataParallel) if needed to get the raw model
        raw_model = self.model.module if cfg.ddp.ddp else self.model

        # The best validation loss encountered so far
        best_val_loss = 1e9

        # Initialize the running memory footprint utility (MFU) as -1.0
        running_mfu = -1.0

        for iter_num in range(cfg.optimizer.max_iters):
            
            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if cfg.learning_rate.decay_lr else cfg.learning_rate.learning_rate

            # Update learning rate in all parameter groups
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            # Train for one epoch
            train_loss = self._train(iter_num=iter_num)

            # Evaluate on validation set at regular intervals and on the first device (if multiple devices are used)
            if iter_num % cfg.io_metrics.eval_interval ==  0 and (not cfg.ddp.ddp or self.device == 0):
                val_loss = self._eval()
                tqdm.write(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}",
                           end='\n' if cfg.ddp.ddp else '\r')

                if cfg.io_metrics.wandb_log:
                    wandb.log({
                        'iter': iter_num,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'lr': lr,
                        'mfu': running_mfu * 100,
                    })

                if val_loss < best_val_loss or cfg.io_metrics.always_save_checkpoint:
                    best_val_loss = val_loss
                    self._save_model(iter_num,
                                     best_val_loss)   

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
                tqdm.write(f"iter {iter_num}: loss {train_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%",
                           end='\n' if cfg.ddp.ddp else '\r')

    def config_to_dict(self):
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

    def _log_build_file_path(self):
        """
        Builds the file path for saving the model state.

        Returns:
            file_path_configs (dict): A dictionary containing the file format and arguments to be used for building the file path.
        """

        file_path_configs = {
            "file_format": cfg.io_metrics.wandb_run_name + '_{}_{}_{}_{}_{}_{}_{}.state',
            "args": (cfg.gpt.block_size, cfg.gpt.vocab_size, cfg.gpt.n_layer,
                        cfg.gpt.n_head, cfg.gpt.n_embd, cfg.gpt.dropout, cfg.gpt.bias)
        }

        return file_path_configs

        


