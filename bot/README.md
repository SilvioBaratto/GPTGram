# GPTGram Telegram Auto-Reply Bot

The GPTGram Auto-Reply Bot is a module built to set up an automatic response system for private Telegram chats using Telethon and GPTGram. It utilizes the GPTGram model to generate responses to messages received in a chat, making for a more interactive and dynamic chat experience.

## Features

- Auto-reply to messages in private Telegram chats.
- Store conversation history for each chat in a text file, aiding the generation of contextually accurate replies.
- User-friendly command-line interface to set up and control the bot.

## Setup

You need to set up a Telegram API app to use this bot. To do so, follow these steps:

1. Log in to your Telegram account through the [Telegram app](https://my.telegram.org/apps).
2. Click on API Development Tools and fill out the form.
3. You will get the `API_ID` and `API_HASH` for your application.

Create a `.env` file in the root directory of the repository and add your Telegram API credentials:

```bash
API_ID=your_api_id
API_HASH=your_api_hash
```

## Usage

The bot will start and listen to incoming private messages. It automatically replies to these messages using the GPTGram model.

In case you want to modify bot configurations such as GPT parameters, IO Metrics configurations, optimizer settings, learning rate configurations, DDP configurations, system settings, or sampling configurations, you can do so using command-line arguments. Please refer to the `arg_parser()` function in the [argparser.py](../GPTGram/argparser.py) script for more details.