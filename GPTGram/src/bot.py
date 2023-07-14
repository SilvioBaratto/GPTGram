import os
import json
from dotenv import load_dotenv
from telethon import TelegramClient, events
from faker import Faker

# Load the environment variables from the .env file
load_dotenv()

api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')

client = TelegramClient('anon', api_id, api_hash)

# Initialize a Faker instance
fake = Faker()

# Create a dictionary to hold the conversations
conversations = {}

# Create a dictionary to track whether each user is typing
typing_users = {}

@client.on(events.NewMessage)
async def my_event_handler(event):
    # Get the user's name from the event
    user_name = event.message.sender.first_name or event.message.sender.username

    # Wait until the user has stopped typing before responding
    user_id = event.message.sender_id
    if user_id in typing_users and typing_users[user_id]:
        return

    # Generate a random sentence
    random_text = fake.sentence()

    # Append the user's message to their conversation history
    if user_id not in conversations:
        conversations[user_id] = []
    conversations[user_id].append({
        user_name: event.message.text,
        "Bot": random_text
    })

    # Save the conversation to a file
    with open("conversations.json", "w") as f:
        f.write(json.dumps(conversations))

    # Respond with the random text
    # await event.respond(random_text)

@client.on(events.ChatAction)
async def typing_handler(event):
    # Check if the user has started or stopped typing
    if event.action == 'typing':
        typing_users[event.user_id] = True
    else:
        typing_users[event.user_id] = False

with client:
    client.run_until_disconnected()