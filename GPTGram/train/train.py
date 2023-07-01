import os
import time
import math
import pickle
import inspect
from contextlib import nullcontext
import wandb
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from ..config import Config as cfg
from ..preprocessing import GramDataset
from ..model import GPT

class Trainer:

    def __init__(self, filepath):
        train_file = os.path.join(filepath, 'train.bin')
        val_file = os.path.join(filepath, 'val.bin')

        # check if train.bin and val.bin exist
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            raise FileNotFoundError('train.bin and/or val.bin not found in the provided filepath.')

        # create dataloaders
        self.train_dataloader = self.init_dataloader(train_file)
        self.val_dataloader = self.init_dataloader(val_file)

        self.ddp_init()
        ptdtype = {'float32': torch.float32, 
                   'bfloat16': torch.bfloat16, 
                   'float16': torch.float16
                   }[cfg.system.dtype]
        self.ctx = nullcontext() if cfg.system.device == 'cpu' \
                        else torch.amp.autocast(device_type=cfg.system.device, 
                                                dtype=ptdtype)
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        if cfg.io_metrics.wandb_log and self.master_process:
            wandb.init(project=cfg.io_metrics.wandb_project, 
                       name=cfg.io_metrics.wandb_run_name)

    def ddp_init(self):
        """
        Initialize the Distributed Data Parallel (DDP) training.

        This function first checks whether the environment variable 'RANK' is set or not. The existence
        of 'RANK' indicates that the program is being run in a DDP mode. 

        If the DDP mode is detected, it further initializes the process group for torch.distributed package, 
        sets the current device, determines if the current process is the master process (for logging and 
        checkpointing purposes), sets the seed offset (for random number generation), and calculates the world size
        (i.e., the total number of processes that will be simultaneously training).

        If the DDP mode is not detected (i.e., if the environment variable 'RANK' is not set), it defaults 
        to a single GPU, single process setup.

        The method also sets a deterministic seed for torch's random number generator to ensure the 
        reproducibility of the training process.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
        # Check if the 'RANK' environment variable is set. If it is, we are in a DDP setup.
        self.ddp = int(os.environ.get('RANK', -1)) != -1

        if self.ddp:
            # Initialize the torch.distributed process group with NCCL backend which is the standard for multi-GPU training.
            init_process_group(backend='nccl')

            # Set the distributed training related attributes.
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.device = f'cuda:{self.ddp_local_rank}'

            # Set the current CUDA device to the local rank.
            torch.cuda.set_device(self.device)

            # The process with rank 0 is deemed the master process. This is a common practice and helps manage logging,
            # checkpointing and other similar tasks which you only want performed once.
            self.master_process = self.ddp_rank == 0

            # Set the seed offset to the rank of the process. Each process gets a different seed.
            self.seed_offset = self.ddp_rank

            # Set the world size to the number of processes participating in the distributed training.
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
        else:
            # If not in DDP mode, we default to a single process, single GPU setup.
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Seed the random number generator for torch to ensure reproducibility.
        torch.manual_seed(1337 + self.seed_offset)

    def init_scaler(self):
        """
        Initialize the scaler for mixed precision training.

        This method first checks if the device type is CUDA and if the scaler is available.
        If both conditions are met, it initializes the scaler with the default settings.

        Args:
            None

        Returns:
            torch.cuda.amp.GradScaler: The initialized scaler.

        Examples:
            >>> scaler = self.init_scaler()
        """
        # Check if the device type is CUDA and if the scaler is available
        if cfg.system.device == 'cuda' and torch.cuda.amp is not None:
            # Initialize the scaler with the default settings
            scaler = torch.cuda.amp.GradScaler(enabled=(cfg.system.dtype == 'float16'))
        else:
            # If the device type is not CUDA or the scaler is not available, return a null context
            scaler = nullcontext()

        return scaler

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
        # Get all parameters of the model that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Group the parameters based on their dimensionality
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]  # 2D parameters will have weight decay
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]  # non-2D parameters will not have weight decay

        # Define optimizer groups with different weight decay settings
        optim_groups = [
            {'params': decay_params, 'weight_decay': cfg.optimizer.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Print the number of decayed and non-decayed parameters
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Check if fused AdamW is available and if the device type is CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and cfg.system.device == 'cuda'

        # Define extra arguments for the optimizer
        extra_args = dict(fused=True) if use_fused else dict()

        # Create AdamW optimizer with the given settings
        optimizer = torch.optim.AdamW(optim_groups, 
                                      lr=cfg.optimizer.learning_rate, 
                                      betas=cfg.optimizer.betas, 
                                      **extra_args)

        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
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
        if cfg.system.device == 'cuda' and torch.cuda.amp is not None:
            # Initialize the scaler with the default settings
            scaler = torch.cuda.amp.GradScaler()
        else:
            # If the device type is not CUDA or the scaler is not available, return a null context
            scaler = nullcontext()

        return scaler

    def init_dataloader(self, filepath: str) -> DataLoader:

        batch_size = cfg.data.batch_size
        if cfg.system.use_cuda:
            batch_size *= torch.cuda.device_count()

        dataset = GramDataset(filepath)  # Create a dataset instance

        dataloader = DataLoader(  # An off-the-shelf class
            dataset,
            batch_size=batch_size,  # batching is done automatically
            num_workers=cfg.system.num_workers,
            pin_memory=cfg.system.use_cuda,   # Pinned memory transfers to GPU quickly
        )

        return dataloader
    
    def init_model(self):

        # Initialize the model
        model = GPT()

        # Load the pretrained model if specified
        if cfg.io_metrics.init_from == 'resume':
            self._load_model(model)

        elif cfg.io_metrics.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights {cfg.io_metrics.init_from}")
            model = GPT.from_pretrained(cfg.io_metrics.init_from)
        
        if cfg.system.use_cuda:
            print(f"Using CUDA; {torch.cuda.device_count()} devices.")
            if self.ddp and torch.cuda.device_count() > 1:
                model = DDP(model, device_ids=[self.ddp_local_rank])
            model = model.to(self.device)

        return model

    def save_model(self):
        """
        Save the current state of the model to a checkpoint file.

        Returns:
            None
        """

        def build_file_path(file_format: str, *args) -> str:
            return os.path.join(lib_dir, 'save', file_format.format(*args))
        
        # Determine the library directory based on the "cfg.io_metrics.folder" attribute
        if cfg.io_metrics.folder is None:
            lib_dir = os.path.dirname(os.path.realpath(__file__)) # Use the current directory
        else:
            lib_dir = cfg.io_metrics.folder

        # Get the file path configurations from the '_log_build_file_path' method
        file_path_configs = self._log_build_file_path()

        # Build the file path for saving the model
        file_path = build_file_path(file_path_configs['file_format'], *file_path_configs['args'])

        # Create the necessary directory structure for the file path
        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        # Get the current state of the model
        state = self._log_model_state()

        # Save the model state to the file path
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
            return os.path.join(lib_dir, 'save', file_format.format(*args))

        # Determine the library directory based on the "cfg.io_metrics.folder" attribute
        if cfg.io_metrics.folder is None:
            lib_dir = os.path.dirname(os.path.realpath(__file__))  
        else:
            lib_dir = cfg.io_metrics.folder

        # Get the file path configurations from the '_log_build_file_path' method
        file_path_configs = self._log_build_file_path()

        # Build the file path for loading the model
        file_path = build_file_path(file_path_configs['file_format'], *file_path_configs['args'])

        # Check if the file path exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path '{file_path}' does not exist")
        
        print(f"Resuming training from {lib_dir}")

        # Load the state from the file path
        checkpoint = torch.load(file_path, map_location=cfg.system.device)
        checkpoint_model_args = checkpoint['model_args']

        # Force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            cfg.gpt[k] = checkpoint_model_args[k]

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
            optimizer_state = checkpoint.get('optimizer_state')
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)

    @torch.no_grad()
    def batch_loss(self, batch):
        self.model.eval()
        x_batch, y_batch = batch
        with self.ctx:
            logits, loss = self.model(x_batch, y_batch)
        self.model.train()
        return loss.item()

    def train(self) -> float:
        """
        Trains the model on the training data for the specified number of epochs.

        Args:
            train_dl (torch.utils.data.DataLoader): The training data.

        Returns:
            train_loss (float): The training loss.
        """
        # Initialize the variable for tracking the training loss
        t0 = time.time()
        # number of iteration in the lifetime of this process
        local_iter_num = 0
        # number of iteration in the current epoch
        iter_num = 0
        raw_model = self.model.module if self.ddp else self.model # unwrap DDP container if needed
        running_mfu = -1.0
        best_val_loss = float('inf')

        while True:
            for x_batch, y_batch in self.train_dataloader:
                # determine and set the learning rate for this iteration
                lr = self.get_lr(iter_num) if cfg.learning_rate.decay_lr else cfg.learning_rate.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # evaluate the loss on train / val sets and write checkpoints
                if iter_num % cfg.io_metrics.eval_interval == 0 and self.master_process:
                    losses = self.batch_loss()
                    print(f"step {iter_num}: train_loss = {losses['train_loss']:.3f}, \
                            val_loss = {losses['val_loss']:.3f}")
                    
                    if cfg.io_metrics.wandb_log:
                        wandb.log({
                            'iter': iter_num,
                            'train_loss': losses['train_loss'], 
                            'val_loss': losses['val_loss'],
                            'lr': lr,
                            'mfu': running_mfu * 100, # log as percentage
                        })

                    if losses['val_loss'] < best_val_loss or cfg.io_metrics.always_save_checkpoint:
                        best_val_loss = losses['val_loss']
                        if iter_num > 0:
                            checkpoint = {
                                'model': raw_model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'model_args': {
                                    'n_layer': cfg.gpt.n_layer,
                                    'n_head': cfg.gpt.n_head,
                                    'n_embd': cfg.gpt.n_embd,
                                    'block_size': cfg.data.block_size,
                                    'bias': cfg.gpt.bias,
                                    'vocab_size': cfg.gpt.vocab_size,
                                },
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss
                            }
                            print(f"Saving checkpoint to {cfg.io_metrics.out_dir}")
                            torch.save(checkpoint, os.path.join(cfg.io_metrics.out_dir, 'checkpoint.pt'))

                if iter_num == 0 and cfg.io_metrics.eval_only:
                    return best_val_loss

                # forward backward update, with optional gradiend accumulation to simulate larger batch_size
                # and using the GradScaler if data type is float16
                for micro_step in range(cfg.data.gradient_accumulation_steps):
                    if self.ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but 
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        self.model.require_backward_grad_sync = (micro_step == cfg.data.gradient_accumulation_steps - 1)
                    with self.ctx:
                        logits, loss = self.model(x_batch, y_batch)
                        loss = loss / cfg.data.gradient_accumulation_steps
                    
                    # backward pass, with gradient scaling if training in fp16 mode
                    self.scaler.scale(loss).backward()
                
                # gradient clipping
                if cfg.optimizer.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)

                # optimizer step and scaler in case of fp16
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # flush the gradients as soos as we can, no need for this memory anymore
                self.optimizer.zero_grad(set_to_none=True)

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                if iter_num % cfg.io_metrics.log_interval == 0 and self.master_process:
                    # get loss as float. note this is CPU/GPU sync point
                    # scale up to undo the division above, approximating the true total loss
                    lossf = loss.item() * cfg.data.gradient_accumulation_steps
                    if local_iter_num >= 5: # let the training loop settle a bit
                        mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.data.gradient_accumulation_steps, dt)
                        running_mfu = mfu if running_mfu == -1 else 0.9 * running_mfu + 0.1 * mfu
                    
                    print(f"step {iter_num}: train_loss = {lossf:.4f}, lr = {lr:.3e}, \
                            time = {dt*1000:.4f}, mfu = {running_mfu * 100:.2f}%")
                
                # increment the iteration number
                iter_num += 1
                local_iter_num += 1

            val_losses = []
            for x_val, y_val in self.val_dataloader:
                val_loss = self.batch_loss(x_val, y_val)
                val_losses.append(val_loss)
            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss}")

                # check if we've trained long enough
            if iter_num >= cfg.optimizer.max_iters:
                break
        
        if self.ddp:
            destroy_process_group()
        
        return best_val_loss

    @staticmethod
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
        # 1) linear warmup for warmup_iters steps
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

        


