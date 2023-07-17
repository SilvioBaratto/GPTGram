# Training a model with GPTGram

Training a model using GPTGram involves using the `torchrun` command alongside several flags that dictate the characteristics and behavior of the model. These flags are parsed using the `arg_parser` function provided by GPTGram. Here, we will discuss the purpose of these flags and how to use them.

The general command structure is as follows:

```bash
torchrun --standalone --nproc_per_node=$ngpus train.py $torchrun_flags
```

Where:
- `--standalone` allows running the application in standalone (single-node) mode.
- `--nproc_per_node` specifies the number of GPUs you want to use for training.
- `train.py` is the main script that performs the training of your GPT model.
- `$torchrun_flags` are the flags from the `arg_parser`.

Below are the explanations of the different flags available:

## GPT Config
These are hyperparameters for the transformer model. 

- `--block_size` specifies the number of tokens that can be processed at once.
- `--vocab_size` sets the size of the vocabulary, essentially determining how many unique tokens the model can learn.
- `--n_layer`, `--n_head`, `--n_embd` represent the number of transformer layers, heads, and the size of the embeddings respectively.
- `--dropout` sets the rate of dropout in the model.
- `--bias` controls whether to use bias in Linears and LayerNorms.

## IOMetrics Config
These flags are used for model checkpointing, logging, and evaluation.

- `--out_dir` sets the directory where the model checkpoints and logs will be stored.
- `--eval_interval` and `--log_interval` determine the frequency of model evaluations and logs respectively.
- `--eval_only` allows you to only evaluate the model without further training.
- `--always_save_checkpoint` determines whether a checkpoint should be saved at every epoch.
- `--init_from` specifies the path to a model checkpoint to initialize from.
- `--log` enables logging.
- `--model` sets the model name.
- `--folder` sets the name of the folder where logs will be stored.

## Data Config
These flags are related to the management of the training data.

- `--gradient_accumulation_steps` represents the number of steps for gradient accumulation.
- `--batch_size` sets the number of samples per batch.

## Optimizer Config
These flags control the optimizer used during training.

- `--max_iters` sets the maximum number of training iterations.
- `--weight_decay` applies weight decay (also known as L2 regularization).
- `--beta1` and `--beta2` are hyperparameters for the Adam optimizer.
- `--grad_clip` sets a limit for the gradient to prevent exploding gradients.

## Learning Rate Config
These flags control the learning rate during training.

- `--learning_rate` sets the initial learning rate.
- `--decay_lr` enables learning rate decay over training iterations.
- `--warmup_iters` sets the number of warmup iterations where the learning rate will linearly increase.
- `--lr_decay_iters` specifies the number of iterations over which the learning rate will be decayed.
- `--min_lr` sets the minimum learning rate.

## DDP Config
These flags are used for distributed data parallel training.

- `--backend` sets the backend to be used for distributed training.

## System Config
These flags are used to control system-level settings.

- `--use_cuda` specifies whether to use CUDA for training.
- `--dtype` sets the data type for the tensors.
- `--compile` enables script compilation.
- `--num_workers` sets the number of workers for data loading.

Here's an example of how to use these flags:

```bash
torchrun --standalone --nproc_per_node=8 train.py --init_from=resume --folder=--folder=/path/to/your/model/checkpoint --learning_rate 0.001 --max_iters 50
```

This will train the model with a block size of 1024, 12 layers, 12 heads, an embedding size of 768, a learning rate of 0.001, and a maximum of 50,000 iterations.

Remember to replace the values based on your own requirements. Always keep in mind the computational resources you have available.

# Sampling a model with GPTGram

Once you have trained a model, you can generate or sample new data from it. The sampling process in GPTGram is performed by executing the `sample.py` script alongside several flags, as in the training phase. The command structure for sampling is:

```bash
python3 sample.py $torchrun_flags
```

Where:
- `sample.py` is the main script that performs the sampling from your GPT model.
- `$torchrun_flags` are the flags from the `arg_parser` related to the sampling process.

Below are the explanations of the sampling-related flags available:

## Sampling Config
These flags are used for sampling from the model.

- `--start` sets the start token for sampling. This is the initial seed for the text generation.
- `--num_samples` sets the number of samples to generate. The model will generate this number of independent text samples.
- `--max_new_tokens` sets the maximum number of new tokens that can be generated per step. This is to limit the length of generated text.
- `--temperature` sets the temperature for sampling. Higher values increase randomness in the generation process, while lower values make it more deterministic.
- `--top_k` sets the number of top alternatives to consider for each step. This parameter can be used to influence the diversity of the generated text.
- `--seed` sets the seed for random number generation. Setting this can ensure that your sampling results are reproducible.

Remember, you should also use the `--init_from` and `--folder` flags to specify the model checkpoint you want to sample from and the directory.

Here's an example of how to use these flags:

```bash
python3 sample.py --start "Once upon a time" --max_new_tokens 100 --temperature 0.8 --top_k 20 --init_from=gpt2-xl 
```

This will generate 5 independent samples, each up to 100 tokens long, starting with the phrase "Once upon a time". The sampling will use a temperature of 0.8 and consider the top 20 alternatives at each step. The samples will be generated from the model specified by the path in the `--init_from` flag.

Be sure to replace the placeholder path with the path to your model checkpoint file. You can adjust the other parameters to fine-tune the characteristics of your generated text.

Note: As always, remember that the results of the sampling process highly depend on your trained model, so different models may produce different quality samples.