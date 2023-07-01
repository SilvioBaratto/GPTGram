import argparse
import torch
from config import Config as cfg

def arg_parser():
    parser = argparse.ArgumentParser(description='GPT Configuration')
    parser.add_argument('--block_size', type=int, default=1024, help='Block size')
    parser.add_argument('--vocab_size', type=int, default=50304, help='Vocabulary size')
    parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=12, help='Number of heads')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', type=bool, default=True, help='Use bias in Linears and LayerNorms')

    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--eval_interval', type=int, default=2000, help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations for evaluation')
    parser.add_argument('--eval_only', type=bool, default=False, help='Evaluate only')
    parser.add_argument('--always_save_checkpoint', type=bool, default=True, help='Always save checkpoint')
    parser.add_argument('--init_from', type=str, default='scratch', help='Initialize from')
    parser.add_argument('--wandb_log', type=bool, default=False, help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='owt', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='gpt2', help='Wandb run name')

    parser.add_argument('--dataset', type=str, default='whatsdataset', help='Dataset name')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40, help='Gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')

    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=600000, help='Maximum iterations')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta 1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Beta 2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clip')

    parser.add_argument('--decay_lr', type=bool, default=True, help='Decay learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--lr_decay_iters', type=int, default=600000, help='Learning rate decay iterations')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')

    parser.add_argument('--backend', type=str, default='nccl', help='DDP backend')

    parser.add_argument('--use_cuda', type=bool, default=torch.cuda.is_available(), help='Use CUDA')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Data type')
    parser.add_argument('--compile', type=bool, default=True, help='Compile')

    parser.add_argument('--start', type=str, default='\n', help='Start token')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum new tokens')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature')
    parser.add_argument('--top_k', type=int, default=200, help='Top k')
    parser.add_argument('--seed', type=int, default=1337, help='Seed')

    args = parser.parse_args()

    # Set values for GPTConfig
    for key, value in vars(args).items():
        setattr(cfg.gpt, key, value)

    # Set values for IOMetricsConfig
    for key, value in vars(args).items():
        setattr(cfg.io_metrics, key, value)

    # Set values for DataConfig
    for key, value in vars(args).items():
        setattr(cfg.data, key, value)

    # Set values for ModelConfig
    for key, value in vars(args).items():
        setattr(cfg.model, key, value)

    # Set values for OptimizerConfig
    for key, value in vars(args).items():
        setattr(cfg.optimizer, key, value)

    # Set values for LearningRateConfig
    for key, value in vars(args).items():
        setattr(cfg.learning_rate, key, value)

    # Set values for DDPConfig
    for key, value in vars(args).items():
        setattr(cfg.ddp, key, value)

    # Set values for SystemConfig
    for key, value in vars(args).items():
        setattr(cfg.system, key, value)

    # Set values for SamplingConfig
    for key, value in vars(args).items():
        setattr(cfg.sampling, key, value)

def 

if __name__ == '__main__':
    pass
