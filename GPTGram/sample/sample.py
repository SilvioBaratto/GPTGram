"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from ..model import GPT
from ..config import Config as cfg

class GramSampler:
    """
    A class for sampling from a trained GPT model.

    This class is used to sample from a trained GPT model. It is initialized with a model and a tokenizer, and
    provides a sample method that can be used to generate text from the model.
    
    """

    def __init__(self, **kwargs):
        pass

    def _init_ctx(self):
        ptdtype = {'float32': torch.float32,
                   'bfloat16': torch.bfloat16,
                   'float16': torch.float16}[cfg.system.dtype]
        
        self.ctx = nullcontext() if cfg.system.use_cuda else torch.amp.autocast(device_type='cuda', 
                                                                                dtype=ptdtype)

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

        # If specified in the configuration, resume training from a checkpoint
        if cfg.io_metrics.init_from == 'resume':
            self._load_model(model)

        # Alternatively, If specified in the configuration, initialize from a pretrained model with
        # pretrained GPT-2 weights
        elif cfg.io_metrics.init_from.startswith('gpt2'):
            print(f"Initializing from OpenAI GPT-2 weights {cfg.io_metrics.init_from}")
            model = GPT.from_pretrained(cfg.io_metrics.init_from)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
        # Move the model to the appropriate device
        self.model = model.to(self.device)

        if cfg.system.compile:
            self.model = torch.compile(self.model)

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

    def _load_metadata(self):
        load_meta = False
        if cfg.io_metrics.init_from == 'resume': 
            meta_path = os.path.join('data', self.checkpoint['config']['dataset'], 'meta.pkl')
            load_meta = os.path.exists(meta_path)
        if load_meta:
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)
            self.stoi, self.itos = self.meta['stoi'], self.meta['itos']
            self.encode = lambda s: [self.stoi[c] for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            print("No meta.pkl found, assuming GPT-2 encodings...")
            self.enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda s: self.enc.encode(s, allowed_special={""})
            self.decode = lambda l: self.enc.decode(l)

    def _encode_prompt(self, start):
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()
        start_ids = self.encode(start)
        self.x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

    def generate(self, num_samples, max_new_tokens, temperature, top_k):
        with torch.no_grad():
            with self.ctx:
                for k in range(num_samples):
                    self.y = self.model.generate(self.x, max_new_tokens, temperature=temperature, top_k=top_k)
                    print(self.decode(self.y[0].tolist()))
                    print('---------------')
