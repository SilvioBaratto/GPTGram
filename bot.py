from telethon import TelegramClient, events

# Use your own values here
api_id = '20394521'
api_hash = '77842b3428bbd170f5b74f931edb6c11'

client = TelegramClient('anon', api_id, api_hash)

@client.on(events.NewMessage)
async def my_event_handler(event):
    # Echo the user message
    await event.respond(event.message.text)

with client:
    client.run_until_disconnected()