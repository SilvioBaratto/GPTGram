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
GPTGram provides a script named `prepare.py` located in the `data` directory to facilitate chat data preparation for training. It processes text files, filters specified phrases, removes timestamps, and encodes the text using Byte-Pair Encoding (BPE). The processed tokens are then written to a binary output file.

To use this:

1. Ensure chat data files are located in a specific directory. By default, it looks for `.txt` files in the `chats/` directory and its subdirectories. You can specify a different directory with the `--folder` argument.

2. Run the script with the following command:
```bash
python prepare.py --folder <data_directory>
```
Replace `<data_directory>` with the path to your chat data.

3. The script generates a `train.bin` and `val.bin` file containing the Byte-Pair Encoded tokens of your chat data.

Note: The script assumes the input chat files have a specific format where timestamps appear like `[12/12/12, 12:12:12]`. If your files are formatted differently, modify the script to match your format. The `prepare.py` script uses Byte-Pair Encoding (BPE) for tokenization and `tiktoken` library for tokenization.

## Training
GPTGram allows you to train the model on your own data. A convenient shell script named `train.sh` simplifies the process of job submission to an HPC cluster using Slurm workload manager. This script will prompt you to enter computational resources requirements, generate a temporary Slurm job submission script, and submit it.

To learn more about this script and its usage, refer to its dedicated [README.md](cluster/README.md) file. For a complete explanation of the usage of the `train.py` script, refer to its dedicated [README.md](cmd/README.md) file.

## Inference
Once the model is trained, you can utilize it to generate responses. The Telegram API integrates with the model, creating a functional chatbot. Instructions to adjust bot configurations, such as GPT parameters, IO Metrics configurations, optimizer settings, learning rate configurations, DDP configurations, system settings, or sampling configurations, are provided in the `arg_parser()` function located in the [argparser.py](GPTGram/argparser.py) file.

## Contributing
Contributions to the GPTGram project are welcomed. Feel free to fork the repository and submit pull requests.

Remember to replace `<repository_directory>` with the directory name of your actual repository.
