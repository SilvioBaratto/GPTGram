import torch
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster.

@dataclass
class IOMetricsConfig:
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    wandb_log: bool = False
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2'

@dataclass
class DataConfig:
    dataset: str = 'whatsdataset'
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    block_size: int = 1024

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

@dataclass
class OptimizerConfig:
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

@dataclass
class LearningRateConfig:
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

@dataclass
class DDPConfig:
    backend: str = 'nccl'

@dataclass
class SystemConfig:
    use_cuda: bool = torch.cuda.is_available()
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: str = 'bfloat16'
    compile: bool = True

@dataclass
class SamplingConfig:
    start: str = "\n"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    seed: int = 1337

@dataclass
class Config:
    gpt: GPTConfig = GPTConfig()
    io_metrics: IOMetricsConfig = IOMetricsConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    learning_rate: LearningRateConfig = LearningRateConfig()
    ddp: DDPConfig = DDPConfig()
    system: SystemConfig = SystemConfig()
    sampling: SamplingConfig = SamplingConfig()
    
    
    
    