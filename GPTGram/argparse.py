import argparse
from GPTGram.config import Config as cfg

def arg_parser():
    """
    Defines and parses command line arguments for GPTGram's configuration.

    Args:
        None

    Returns:
        args: Namespace object that contains the values of command line arguments. 

    Example Usage:
        >>> args = arg_parser()
        >>> print(args.block_size)
        512

    """

    parser = argparse.ArgumentParser(description='GPT Configuration')

    # GPT Config
    parser.add_argument('--block_size', type=int, default=cfg.gpt.block_size, 
                        help='The maximum length of a sequence for the GPT model. '
                             'It determines the window size for context in Transformer architecture.')
    parser.add_argument('--vocab_size', type=int, default=cfg.gpt.vocab_size, 
                        help='The size of vocabulary, i.e., the number of unique tokens recognized by the model.')
    parser.add_argument('--n_layer', type=int, default=cfg.gpt.n_layer, 
                        help='The number of transformer layers in the GPT model.')
    parser.add_argument('--n_head', type=int, default=cfg.gpt.n_head, 
                        help='The number of attention heads in the Transformer architecture.')
    parser.add_argument('--n_embd', type=int, default=cfg.gpt.n_embd, 
                        help='The dimension of the embeddings for tokens and positional encodings in the GPT model.')
    parser.add_argument('--dropout', type=float, default=cfg.gpt.dropout, 
                        help='The dropout probability used in the Transformer layers.')
    parser.add_argument('--bias', type=bool, default=cfg.gpt.bias, 
                        help='Boolean indicating if bias should be used in the linear layers and layer normalization.')

    # IOMetrics Config
    parser.add_argument('--out_dir', type=str, default=cfg.io_metrics.out_dir, 
                        help='The directory where the training outputs will be saved.')
    parser.add_argument('--eval_interval', type=int, default=cfg.io_metrics.eval_interval, 
                        help='The interval (in terms of steps) at which the model is evaluated during training.')
    parser.add_argument('--log_interval', type=int, default=cfg.io_metrics.log_interval, 
                        help='The interval (in terms of steps) at which the training metrics are logged.')
    parser.add_argument('--eval_only', action='store_true', default=cfg.io_metrics.eval_only, 
                        help='If set, the script will only perform evaluation without further training.')
    parser.add_argument('--always_save_checkpoint', action='store_true', default=cfg.io_metrics.always_save_checkpoint, 
                        help='If set, the script will save a checkpoint after every evaluation step.')
    parser.add_argument('--init_from', type=str, default=cfg.io_metrics.init_from, 
                        help='Path to initialize model from a pre-trained checkpoint.')
    parser.add_argument('--log', action='store_true', default=cfg.io_metrics.log, 
                        help='If set, enables logging.')
    parser.add_argument('--model', type=str, default=cfg.io_metrics.model, 
                        help='Name of the model for saving and loading purposes.')
    parser.add_argument('--folder', type=str, default=cfg.io_metrics.folder, 
                        help='Name of the folder to save outputs and checkpoints.')

    # Data Config
    parser.add_argument('--gradient_accumulation_steps', type=int, default=cfg.data.gradient_accumulation_steps, 
                        help='The number of steps for gradient accumulation. '
                             'Effectively increases the batch size while using the same amount of memory.')
    parser.add_argument('--batch_size', type=int, default=cfg.data.batch_size, 
                        help='The number of examples in a batch for training.')

    # Optimizer Config
    parser.add_argument('--max_iters', type=int, default=cfg.optimizer.max_iters, 
                        help='The maximum number of training iterations.')
    parser.add_argument('--weight_decay', type=float, default=cfg.optimizer.weight_decay, 
                        help='The weight decay (L2 penalty) used in the optimizer.')
    parser.add_argument('--beta1', type=float, default=cfg.optimizer.beta1, 
                        help='The beta1 parameter for the Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=cfg.optimizer.beta2, 
                        help='The beta2 parameter for the Adam optimizer.')
    parser.add_argument('--grad_clip', type=float, default=cfg.optimizer.grad_clip, 
                        help='The maximum gradient norm for gradient clipping.')

    # Learning Rate Config
    parser.add_argument('--learning_rate', type=float, default=cfg.learning_rate.learning_rate, 
                        help='The initial learning rate for training.')
    parser.add_argument('--decay_lr', action='store_true', default=cfg.learning_rate.decay_lr, 
                        help='If set, enables learning rate decay.')
    parser.add_argument('--warmup_iters', type=int, default=cfg.learning_rate.warmup_iters, 
                        help='The number of warmup iterations for the learning rate scheduler.')
    parser.add_argument('--lr_decay_iters', type=int, default=cfg.learning_rate.lr_decay_iters, 
                        help='The number of iterations over which the learning rate is decayed.')
    parser.add_argument('--min_lr', type=float, default=cfg.learning_rate.min_lr, 
                        help='The minimum learning rate during decay.')

    # DDP Config
    parser.add_argument('--backend', type=str, default=cfg.ddp.backend, 
                        help='The backend to use for distributed data parallel (DDP).')

    # System Config
    parser.add_argument('--use_cuda', action='store_true', default=cfg.system.use_cuda, 
                        help='If set, enables training on CUDA-enabled GPUs.')
    parser.add_argument('--dtype', type=str, default=cfg.system.dtype, 
                        help='The data type to be used in computations (e.g., "float32", "float16").')
    parser.add_argument('--compile', action='store_true', default=cfg.system.compile, 
                        help='If set, compiles the model graph for improved performance.')
    parser.add_argument('--num_workers', type=int, default=cfg.system.num_workers, 
                        help='The number of worker threads to use for data loading.')

    # Sampling Config
    parser.add_argument('--start', type=str, default=cfg.sampling.start, 
                        help='The initial token to start the text generation/sampling.')
    parser.add_argument('--num_samples', type=int, default=cfg.sampling.num_samples, 
                        help='The number of samples/texts to generate.')
    parser.add_argument('--max_new_tokens', type=int, default=cfg.sampling.max_new_tokens, 
                        help='The maximum number of new tokens to generate in each step.')
    parser.add_argument('--temperature', type=float, default=cfg.sampling.temperature, 
                        help='The temperature used in sampling. Higher values increase randomness.')
    parser.add_argument('--top_k', type=int, default=cfg.sampling.top_k, 
                        help='The number of top alternatives to consider in each step of sampling.')
    parser.add_argument('--seed', type=int, default=cfg.sampling.seed, 
                        help='The random seed used for reproducibility.')

    args = parser.parse_args()

    return args

