import os
import csv
import math
import torch
import time
import itertools
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from ..preprocessing import GramDataset
from ..config import Config as cfg
from ..base import BaseGram

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
            fieldnames = ['train_loss', 'val_loss', 'eval_time', 'lr']
            log_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            log_writer.writeheader()

    # Write the log data to the file
    with open(filename, 'a', newline='') as csv_file:
        log_writer = csv.DictWriter(csv_file, fieldnames=log_data.keys())
        log_writer.writerow(log_data)

        
class GramTrainer(BaseGram):
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

    def __init__(self, **kwargs):
        """
        Initializes the trainer with the file path for the train and validation data, the model configuration,
        the model, the optimizer, and the scaler.

        Args:
            filepath: The directory path of the binary data files 'train.bin' and 'val.bin'.
            **kwargs: Additional keyword arguments for initializing the model configuration.
        """
        super().__init__(**kwargs)
        self._init_paths()

        self.train_dataloader = self.init_dataloader(self.train_file)
        self.val_dataloader = self.init_dataloader(self.val_file)

        if cfg.system.walltime is not None:
            h, m, s = map(int, cfg.system.walltime.split(':'))
            self.wall_time = h * 3600 + m * 60 + s
        # Get the time at the start of the job.
        self.start_time = time.time()

    def _init_paths(self) -> None:
        """
        Initializes the paths for the training and validation data files. and checks if they exist.

        Args:
            filepath: The directory path of the binary data files 'train.bin' and 'val.bin'.
        """

        self.train_file = os.path.join(cfg.io_metrics.dataset, 'train.bin')
        self.val_file = os.path.join(cfg.io_metrics.dataset, 'val.bin')

        for file in [self.train_file, self.val_file]:
            if not os.path.exists(file):
                raise FileNotFoundError(f'{file} not found.')
            
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
        best_val_loss = self.best_val_loss if hasattr(self, 'best_val_loss') else 1e9

        # Initialize the variable for tracking the total loss
        total_loss = 0.0

        # evaluation time
        eval_time = time.time()

        # Iterate over all batches in the training data loader
        start_iter = self.iter_num if hasattr(self, 'iter_num') else 0

        # Initialize warmup factor
        warmup_iters = int(len(self.train_dataloader) * 0.1)

        # Initialize lr scheduler
        lr_decay_iters = int(len(self.train_dataloader) * 0.9)

        # Initialize scheduler
        scheduler = CosineAnnealingLR(self.optimizer, 
                                    T_max=lr_decay_iters, 
                                    eta_min=cfg.learning_rate.min_lr)
        
        # Iterate over all batches in the training data loader
        for idx, (x_batch, y_batch) in enumerate(itertools.islice(self.train_dataloader, start_iter, None)):
            i = idx + start_iter
            if cfg.system.is_slurm:
                elapsed_time = time.time() - self.start_time
                # save the model and finish the loop ten minutes before walltime
                if self.wall_time - elapsed_time <= 10 * 60: 
                    self._save_model(i, best_val_loss)
                    # Break the loop and end the job to avoid SLURM termination
                    break  

            # Check if CUDA is available and if it is, use it and pin memory for faster CPU-to-GPU transfer
            x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            # warmup phase: linear scaling of learning rate
            if not cfg.learning_rate.decay_lr or i < warmup_iters:
                warmup_factor = i / warmup_iters if i < warmup_iters else 1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_factor * cfg.learning_rate.learning_rate

            self.model.require_backward_grad_sync = (i+1) % cfg.data.gradient_accumulation_steps == 0 if cfg.ddp.ddp else True

            # Perform a forward pass through the model and compute the loss
            with self.ctx:
                logits, loss = self.model(x_batch, y_batch)
                loss = loss / cfg.data.gradient_accumulation_steps

            if isinstance(self.scaler, torch.cuda.amp.GradScaler):
                # If the scaler is a GradScaler, we can use it for mixed-precision training
                loss_scaled = self.scaler.scale(loss)
                loss_scaled.backward()
            else:
                # If the scaler is a nullcontext, we're not doing mixed-precision training
                loss.backward()

            total_loss += loss.item()

            if ((i+1) % cfg.data.gradient_accumulation_steps == 0) or (i + 1 == len(self.train_dataloader)):
                # Clip the gradients if a threshold is specified in the configuration
                if cfg.optimizer.grad_clip != 0:
                    if isinstance(self.scaler, torch.cuda.amp.GradScaler):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optimizer.grad_clip)

                # Update the model parameters
                if isinstance(self.scaler, torch.cuda.amp.GradScaler):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Update the model parameters without scaling if CUDA is not available
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # Update learning rate using the scheduler after optimizer.step()
            if cfg.learning_rate.decay_lr and i >= warmup_iters:
                scheduler.step()

            # Check if it's time to perform validation
            if time.time() - eval_time >= cfg.io_metrics.save_interval * 60:  
                start_eval_time = time.time()
                val_loss = self._eval()
                end_eval_time = time.time()
                eval_duration = (end_eval_time - start_eval_time) / 60  

                print(f"train loss: {total_loss / (i+1):.4f},\
                    val loss: {val_loss:.4f},\
                    eval time: {eval_duration:.4f}m,\
                    batch {i+1} of {len(self.train_dataloader)}")
                            
                if cfg.io_metrics.log:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_data = {
                        'train_loss': total_loss / (i+1),
                        'val_loss': val_loss,
                        'eval_time': eval_duration,
                        'lr': lr,
                    }

                    log_to_csv('training_log.csv', log_data)

                if val_loss < best_val_loss or cfg.io_metrics.always_save_checkpoint:
                    best_val_loss = val_loss
                    if not cfg.ddp.ddp or self.device == 0:
                        self._save_model(i, best_val_loss)  

                eval_time = time.time() 

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

        # back to training mode
        self.model.train()

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
            
            self.iter_num = 0
