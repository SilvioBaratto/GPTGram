# GPTGram
GPTGram is a project that combines the power of Generative Pretraining Transformers (GPT) with the versatility of Telegram's API to create a responsive and interactive chatbot. It utilizes a model trained on chat data to respond to messages in a human-like manner.

## Installation
You can clone the GPTGram repository and install the package using the following commands:

```bash
> git clone https://github.com/SilvioBaratto/GPTGram
> cd GPTGram
> ./install.sh
```

## Requirements
Before using the GPTGram Telegram chatbot API, ensure you have installed the dependencies listed in the `requirements.txt` file. Additionally, if you intend to use Flash Attention instead of slow attention, you are recommended to install PyTorch 2.0. Depending on your CUDA version, follow the respective steps:

- For CUDA 11.8:
```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
```

- For CUDA 11.7:
```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```

- For CPU only:
```bash
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
```

Next, install the remaining dependencies:

```bash
pip install -r requirements.txt
```
## Preparing Chat Data for Training

Training GPTGram requires conversational data, and the provided `prepare.py` script helps to facilitate this process. This script is responsible for processing the input text files, filtering specific phrases, removing timestamps, and encoding the text using Byte-Pair Encoding (BPE). The output is a binary file containing the processed tokens, which is ready to be consumed by the training script.

Here's how I prepared my data:

1. **Exporting Chats:** From the WhatsApp mobile application, I exported individual chats. Each chat was exported as a `.txt` file and stored in a directory named `data/`. Ensure all your chat data files are placed in a specific directory.

2. **Preprocessing Data:** I used the `prepare.py` script to preprocess the concatenated chat file:
    ```bash
    python prepare.py --folder data/
    ```
   Replace `data/` with the path to your directory if you've used a different one.

After running the script, `train.bin` and `val.bin` files will be generated in the same directory. These files contain the Byte-Pair Encoded tokens of your chat data, ready to be used for training.

Please note, the `prepare.py` script assumes that your chat data is formatted in a certain way. Specifically, it looks for timestamps formatted like `[12/12/12, 12:12:12]`. If your data is formatted differently, you'll need to modify the script to match your specific formatting.

The `prepare.py` script uses Byte-Pair Encoding (BPE) for tokenization and the `tiktoken` library from OpenAI. Make sure you have these dependencies installed before you run the script.

## Training
GPTGram allows you to train the model on your own data. A convenient shell script named `train.sh` simplifies the process of job submission to an HPC cluster using Slurm workload manager. This script will prompt you to enter computational resources requirements, generate a temporary Slurm job submission script, and submit it.

To learn more about this script and its usage, refer to its dedicated [README.md](cluster/README.md) file. For a complete explanation of the usage of the `train.py` script, refer to its dedicated [README.md](cmd/README.md) file.

## Sampling
After you've trained your GPTGram model, you can generate samples from it using the `sample.py` script. This allows you to see the kind of output your model generates, which can be particularly useful for tuning or debugging.

Please note that the `sample.py` script will use the configuration that was active during training. If you wish to modify the sampling configuration, such as the start token, the number of samples, or the temperature, you will need to edit the corresponding values in the `config.py` file before running the script.

Refer to the documentation for the [config.py](cmd/README.md) file for a full list of configurable sampling parameters and their descriptions.

## Inference
Once the model is trained, you can utilize it to generate responses. The Telegram API integrates with the model, creating a functional chatbot. Instructions to adjust bot configurations, such as GPT parameters, IO Metrics configurations, optimizer settings, learning rate configurations, DDP configurations, system settings, or sampling configurations, are provided in the `arg_parser()` function located in the [argparser.py](GPTGram/argparser.py) file.

## Contributing
Contributions to the GPTGram project are welcomed. Feel free to fork the repository and submit pull requests.

Remember to replace `<repository_directory>` with the directory name of your actual repository.
