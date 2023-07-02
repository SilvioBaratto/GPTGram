# GPTGram
Telegram chatbot API using generative pretraining transformers

### Installing from source

You can clone this repository on your local machines using:

```bash
> git clone https://github.com/SilvioBaratto/GPTGram
```

To install the package:

```bash
> cd GPTGram
> ./install.sh
```

---

# Prepare Chat Data for Training

This repository contains a Python script `prepare.py` in `data` folder that helps prepare chat data for training a language model. The script reads chat data from text files, filters out specified phrases (like "image omitted", "sticker omitted", etc.), removes timestamps, and encodes the text using Byte-Pair Encoding (BPE). The encoded tokens are then written to a binary output file.

### Installation Requirements

Before using the GPTGram Telegram chatbot API, make sure to install the required dependencies listed in the `requirements.txt` file. Additionally, if you want to use Flash Attention instead of slow attention, it is recommended to install PyTorch 2.0 from the following sources based on your CUDA version:

- If CUDA 11.8 is installed, run the following command:

  ```bash
  pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
  ```

- If CUDA 11.7 is installed, run the following command:

  ```bash
  pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
  ```

- If you are using CPU only, run the following command:

  ```bash
  pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cpu
  ```

These commands will install the necessary versions of NumPy, Torch, TorchVision, and TorchAudio based on your CUDA configuration.

Finally, you can install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

This will install all the required Python libraries and dependencies needed to run GPTGram successfully.

---

## How to Use

1. The script expects chat data files to be located in a specific directory. By default, it looks for `.txt` files in the `chats/` directory and its subdirectories. If your data is located elsewhere, you can specify the directory with the `--folder` argument.

2. To run the script, use the following command:

    ```bash
    python prepare.py --folder <data_directory>
    ```

    Replace `<data_directory>` with the path to your chat data. For example, if your data is in a directory called `my_data/`, you would run:

    ```bash
    python prepare.py --folder my_data/
    ```

3. The script will create a binary file `train.bin` in the current directory. This file contains the Byte-Pair Encoded tokens of your chat data.

4. If everything goes well, you should see a message like `data has X tokens`, where X is the total number of tokens in your data.

## Notes

- The script assumes that the input chat files have a specific format, specifically that timestamps appear in a format like `[12/12/12, 12:12:12]`. If your chat files are formatted differently, you will need to modify the script to match your format.
- The `prepare.py` script uses Byte-Pair Encoding (BPE) for tokenization. This is a form of subword tokenization that is used in many modern language models, including GPT-2 and GPT-3.
- This script uses the `tiktoken` library for tokenization. This is an open-source library developed by OpenAI, and it uses the same tokenization method as GPT-2 and GPT-3.

---

Remember to replace `<repository_directory>` with the directory name of your actual repository.
