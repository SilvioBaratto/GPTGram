import torch
from contextlib import nullcontext
from dataclasses import dataclass
import os

@dataclass
class GPTConfig:
    """
    Configuration parameters for the GPT model.

    Attributes:
        block_size (int): The maximum length of a sequence for the GPT model. 
                          Determines the window size for context in Transformer architecture.
                          Default is 1024.
        vocab_size (int): The size of vocabulary, i.e., the number of unique tokens recognized by the model.
                          Default is 50304.
        n_layer (int): The number of transformer layers in the GPT model. Default is 12.
        n_head (int): The number of attention heads in the Transformer architecture. Default is 12.
        n_embd (int): The dimension of the embeddings for tokens and positional encodings in the GPT model.
                      Default is 768.
        dropout (float): The dropout probability used in the Transformer layers. Default is 0.0.
        bias (bool): Boolean indicating if bias should be used in the linear layers and layer normalization.
                     Default is True.
    """

    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

@dataclass
class IOMetricsConfig:
    """
    Configuration parameters for I/O and metric tracking.

    Attributes:
        out_dir (str): The directory where the training outputs will be saved. Default is 'out'.
        eval_interval (int): The interval (in steps) at which the model is evaluated during training. 
                             Default is 3.
        log_interval (int): The interval (in steps) at which the training metrics are logged. Default is 1.
        eval_only (bool): If set to True, the script will only perform evaluation without further training.
                          Default is False.
        always_save_checkpoint (bool): If set to True, the script will save a checkpoint after every evaluation step.
                                       Default is True.
        init_from (str): Path to initialize model from a pre-trained checkpoint. Default is 'scratch'.
        log (bool): If set to True, enables logging. Default is True.
        model (str): Name of the model for saving and loading purposes. Default is 'gpt2'.
        folder (str): Name of the folder to save outputs and checkpoints. Default is '../dataset/'.
    """

    out_dir: str = 'out'
    eval_interval: int = 3
    log_interval: int = 1
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    log: bool = True
    model: str = 'gpt2'
    folder: str = '../dataset/'

@dataclass
class DataConfig:
    """
    Configuration parameters for the data handling during training.

    Attributes:
        gradient_accumulation_steps (int): The number of steps for gradient accumulation. Default is 5 times the number
                                           of CUDA devices if CUDA is available, else 5.
        batch_size (int): The number of examples in a batch for training. Default is 12.
        block_size (int): The maximum length of a sequence for the GPT model. Default is 1024.
    """

    gradient_accumulation_steps: int = 5 * torch.cuda.device_count() if torch.cuda.is_available() else 5
    batch_size: int = 12
    block_size: int = 1024

@dataclass
class OptimizerConfig:
    """
    Configuration parameters for the optimizer used in training.

    Attributes:
        max_iters (int): The maximum number of training iterations. Default is 50.
        weight_decay (float): The weight decay (L2 penalty) used in the optimizer. Default is 1e-1.
        beta1 (float): The beta1 parameter for the Adam optimizer. Default is 0.9.
        beta2 (float): The beta2 parameter for the Adam optimizer. Default is 0.95.
        betas (tuple): A tuple containing beta1 and beta2 parameters for the Adam optimizer. Default is (0.9, 0.95).
        grad_clip (float): The maximum gradient norm for gradient clipping. Default is 1.0.
    """

    max_iters: int = 50
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    betas: tuple = (beta1, beta2)
    grad_clip: float = 1.0

@dataclass
class LearningRateConfig:
    """
    Configuration parameters for the learning rate used in training.

    Attributes:
        learning_rate (float): The initial learning rate for training. Default is 6e-4.
        decay_lr (bool): If set to True, enables learning rate decay. Default is True.
        warmup_iters (int): The number of warmup iterations for the learning rate scheduler. Default is 10% of max_iters.
        lr_decay_iters (int): The number of iterations over which the learning rate is decayed. Default is 90% of max_iters.
        min_lr (float): The minimum learning rate during decay. Default is 6e-5.
    """

    learning_rate: float = 6e-4
    decay_lr: bool = True
    warmup_iters: int = int(OptimizerConfig.max_iters * 0.1)
    lr_decay_iters: int = int(OptimizerConfig.max_iters * 0.9)
    min_lr: float = 6e-5

@dataclass
class DDPConfig:
    """
    Configuration parameters for Distributed Data Parallel (DDP) training.

    Attributes:
        backend (str): The backend to use for DDP. Default is 'nccl'.
        ddp (bool): If set to True, enables DDP. Default is True if environment variable 'RANK' is not -1, else False.
    """

    backend: str = 'nccl'
    ddp: bool = int(os.environ.get('RANK', -1)) != -1

@dataclass
class SystemConfig:
    """
    Configuration parameters for the system settings during training.

    Attributes:
        use_cuda (bool): If set to True, enables training on CUDA-enabled GPUs. Default is True if CUDA is available, else False.
        device (str): The device to use for training. Default is 'cuda' if CUDA is available, else 'cpu'.
        dtype (str): The data type to be used in computations. Default is 'bfloat16' if CUDA is available and supports bf16, else 'float16'.
        compile (bool): If set to True, compiles the model graph for improved performance. Default is True.
        num_workers (int): The number of worker threads to use for data loading. Default is 'SLURM_CPUS_PER_TASK' environment variable if set, else 4.
    """

    use_cuda: bool = torch.cuda.is_available()
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = True
    num_workers: int = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 4

@dataclass
class SamplingConfig:
    """
    Configuration parameters for the text sampling process.

    Attributes:
        start (str): The initial token to start the text generation/sampling. Default is "\n".
        user (str): The user to be mimicked during text generation. Default is "silvio: ".
        num_samples (int): The number of samples/texts to generate. Default is 10.
        max_new_tokens (int): The maximum number of new tokens to generate in each step. Default is 500.
        temperature (float): The temperature used in sampling. Higher values increase randomness. Default is 0.8.
        top_k (int): The number of top alternatives to consider in each step of sampling. Default is 200.
        seed (int): The random seed used for reproducibility. Default is 1337.
    """

    start: str = "\n"
    user: str = "silvio: "
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    seed: int = 1337

@dataclass
class Config:
    """
    Master configuration object that contains all the configuration classes.

    Attributes:
        gpt (GPTConfig): The GPT model configuration. Default is GPTConfig().
        io_metrics (IOMetricsConfig): The I/O and metrics configuration. Default is IOMetricsConfig().
        data (DataConfig): The data handling configuration. Default is DataConfig().
        optimizer (OptimizerConfig): The optimizer configuration. Default is OptimizerConfig().
        learning_rate (LearningRateConfig): The learning rate configuration. Default is LearningRateConfig().
        ddp (DDPConfig): The DDP configuration. Default is DDPConfig().
        system (SystemConfig): The system settings configuration. Default is SystemConfig().
        sampling (SamplingConfig): The text sampling configuration. Default is SamplingConfig().
    """

    gpt: GPTConfig = GPTConfig()
    io_metrics: IOMetricsConfig = IOMetricsConfig()
    data: DataConfig = DataConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    learning_rate: LearningRateConfig = LearningRateConfig()
    ddp: DDPConfig = DDPConfig()
    system: SystemConfig = SystemConfig()
    sampling: SamplingConfig = SamplingConfig()

    
    
    
    
