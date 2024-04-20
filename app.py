import discord
import requests
import json
import asyncio
from PIL import Image
from io import BytesIO
from discord.ext import commands
import gpt_chat
from gpt_chat import GPTChat
from collections import deque
import numpy as np
import google.generativeai as genai
import random
import string
import os
from discord import File
import subprocess
from discord.ext import commands
import psutil
from bardapi import Bard
MAX_MESSAGE_LENGTH = 2000


pers="" # Bots Persona
help="" # Bots Help Commad
endpoint = "" # Default Endpoint
user_counters = {} 
user_history = {}
genai.configure(api_key="") 
bot= commands.Bot(command_prefix="/", intents=discord.Intents.all())
api_key = ""
users_seen_disclaimer = {}
temperature = 0.5

def setup_database():
  """ 
  Setup the DB
  The bellow is a sample DB and is not recommended for use
  """

    conn = sqlite3.connect('database.db')

    # Create a cursor object
    c = conn.cursor()

    # Create table
    c.execute('''
        CREATE TABLE  IF NOT EXISTS shared_data
        (user_id text, url text, model text, api_key text, key text)
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

setup_database()




async def show_disclaimer(ctx):
    global users_seen_disclaimer
    # Check if the user has seen the disclaimer
    if ctx.author.id in users_seen_disclaimer:
        if users_seen_disclaimer[ctx.author.id] == True:
            return True
    disclaimer_message = """
    Ethical Disclaimer:
This bot, its reverse proxy, and server are designed to facilitate interaction with AI technologies and provide a platform for testing AI APIs. While we strive to ensure the accuracy and reliability of the results returned by the bot, its reverse proxy, and server, we cannot guarantee their correctness, completeness, or suitability for any particular purpose. The bot, its reverse proxy, and server may return results that are unexpected, inaccurate, or objectionable. Users are advised to use their discretion and judgment while interpreting and using the results. We disclaim all liability for any damages arising from the use of this bot, its reverse proxy, and server, or the results they may return. The bot, its reverse proxy, and server are not intended to replace professional advice in any field, including but not limited to, financial, medical, or legal matters. The use of AI models via the bot, its reverse proxy, and server should comply with all applicable laws and regulations, respect intellectual property rights, and not be used for any unlawful or unethical activities. Information collected by the bot, its reverse proxy, and server is not guaranteed to be secure and we disclaim all liability for any breaches or unauthorized access. Users are responsible for protecting their own data and should not share sensitive information with the bot, its reverse proxy, or server.
    """
    # Send the disclaimer message
    message = await ctx.send(disclaimer_message)
    # Add reactions to the message
    await message.add_reaction('✅')
    await message.add_reaction('❌')

    def check(reaction, user):
        return user == ctx.author and str(reaction.emoji) in ['✅', '❌']

    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=60.0, check=check)
    except asyncio.TimeoutError:
        await ctx.send('🔴 No reaction within the time limit!')
        return False
    else:
        if str(reaction.emoji) == '✅':
            await ctx.send('🟢 You agreed to the disclaimer. The program will continue.')
            await asyncio.sleep(4)
            await message.delete()
            # Add the user to the dictionary
            users_seen_disclaimer[ctx.author.id] = True
            return True
        elif str(reaction.emoji) == '❌':
            await ctx.send('🔴 You disagreed with the disclaimer. The program will stop.')
            await asyncio.sleep(3)
            await message.delete()
            return False
    return False

@bot.event
async def on_ready(): 
    print(f'Logged in as {bot.user.name}')




@bot.command()
async def gemini_chat(ctx, *, message):
    # Generate content
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(message)

    # Send the response in the Discord chat
    await ctx.send(response.text)

@bot.command()
async def gemini_image(ctx, *, message):
    # Check if there's an image attached
    if ctx.message.attachments:
        model = genai.GenerativeModel('gemini-pro-vision')
        # Get the URL of the first image attached
        image_url = ctx.message.attachments[0].url

        # Download the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Generate content with an image and text
        response = model.generate_content([message, img])

        # Send the response in the Discord chat
        await ctx.send(response.text)
    else:
        await ctx.send('Please attach an image.')

server_process = None
ngrok_process = None

@bot.command()
async def run_ngrok(ctx):
    global ngrok_process
    # Check if the ngrok is already running
    if ngrok_process is not None and ngrok_process.poll() is None:
        await ctx.send('ngrok is already running.')
        return

    # Run the ngrok server in the background
    try:
        ngrok_process = subprocess.Popen(["ngrok", "http", "5000"])
        await ctx.send('ngrok server is running in the background.')
    except Exception as e:
        await ctx.send(f'Error occurred while starting ngrok: {str(e)}')

@bot.command()
async def proxy(ctx):
    await show_disclaimer(ctx)
  
    # Get the ngrok URL
    try:
        response = requests.get('http://localhost:4040/api/tunnels')
        data = json.loads(response.text)
        url = data['tunnels'][0]['public_url']
        await ctx.send(f'ngrok URL: {url}')
    except Exception as e:
        await ctx.send(f'Error occurred while getting Proxy URL: {str(e)}')


@bot.command()
async def stop_ngrok(ctx):
    global ngrok_process
    # Check if the ngrok is not running
    if ngrok_process is None or ngrok_process.poll() is not None:
        await ctx.send('ngrok is not running.')
        return

    # Stop the ngrok server
    try:
        ngrok_process.send_signal(signal.SIGTERM)
        ngrok_process = None
        await ctx.send('ngrok server has been stopped.')
    except Exception as e:
        await ctx.send(f'Error occurred while stopping ngrok: {str(e)}')


@bot.command()
async def run_server(ctx):
    global server_process
    # Check if the server is already running
    if server_process is not None and server_process in (p.pid for p in psutil.process_iter(['pid'])):
        await ctx.send('Server is already running.')
        return

    # Run the Flask server in the background
    server_process = subprocess.Popen(["python", "api.py"])
    await ctx.send('Flask server is running in the background.')

@bot.command()
async def stop_server(ctx):
    global server_process
    # Check if the server is running
    if server_process is None or server_process not in (p.pid for p in psutil.process_iter(['pid'])):
        await ctx.send('Server is not running.')
        return

    # Terminate the server process
    server_process.terminate()
    server_process = None
    await ctx.send('Flask server has been stopped.')



@bot.command()
async def gen_key(ctx):
    if(not await show_disclaimer(ctx)):
        return
    user_id = str(ctx.author.id)
    # Generate a random key of length 15
    key = ''.join(random.choices(string.ascii_letters + string.digits, k=15))

    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Check if user_id already exists in the database
    c.execute("SELECT * FROM shared_data WHERE user_id=?", (user_id,))
    result = c.fetchone()

    if result is None:
        # If user_id does not exist, insert a new record
        c.execute("INSERT INTO shared_data (user_id, key) VALUES (?, ?)", (user_id, key))
    else:
        # If user_id exists, update the existing record
        c.execute("UPDATE shared_data SET key = ? WHERE user_id = ?", (key, user_id))

    conn.commit()
    conn.close()

    # Send a private message to the user with the key
    await ctx.author.send(f"Your key has been set to {key}.")



@bot.command()
async def endpoint(ctx, endpoint):
    if(not await show_disclaimer(ctx)):
        return
    user_id = str(ctx.author.id)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Check if user_id already exists in the database
    c.execute("SELECT * FROM shared_data WHERE user_id=?", (user_id,))
    result = c.fetchone()

    if result is None:
        # If user_id does not exist, insert a new record
        c.execute("INSERT INTO shared_data (user_id, url) VALUES (?, ?)", (user_id, endpoint))
    else:
        # If user_id exists, update the existing record
        c.execute("UPDATE shared_data SET url = ? WHERE user_id = ?", (endpoint, user_id))

    conn.commit()
    conn.close()
    await ctx.send('Endpoint has been set.')

@bot.command()
async def SetKey(ctx, message):
    if(not await show_disclaimer(ctx)):
        return
    user_id = str(ctx.author.id)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Check if user_id already exists in the database
    c.execute("SELECT * FROM shared_data WHERE user_id=?", (user_id,))
    result = c.fetchone()

    if result is None:
        # If user_id does not exist, insert a new record
        c.execute("INSERT INTO shared_data (user_id, api_key) VALUES (?, ?)", (user_id, message))
    else:
        # If user_id exists, update the existing record
        c.execute("UPDATE shared_data SET api_key = ? WHERE user_id = ?", (message, user_id))

    conn.commit()
    conn.close()
    await ctx.send("API key has been set.")

@bot.command()
async def SetModel(ctx, message):
    if(not await show_disclaimer(ctx)):
        return
    user_id = str(ctx.author.id)
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Check if user_id already exists in the database
    c.execute("SELECT * FROM shared_data WHERE user_id=?", (user_id,))
    result = c.fetchone()

    if result is None:
        # If user_id does not exist, insert a new record
        c.execute("INSERT INTO shared_data (user_id, model) VALUES (?, ?)", (user_id, message))
    else:
        # If user_id exists, update the existing record
        c.execute("UPDATE shared_data SET model = ? WHERE user_id = ?", (message, user_id))

    conn.commit()
    conn.close()
    await ctx.send(f"The model has been set to {message}")

@bot.command()
async def chat(ctx, *, message):
    if(not await show_disclaimer(ctx)):
        return
    global user_counters
    global user_history
    user_id = ctx.message.author.id
    print(user_id)
    if user_id not in user_counters:
        user_counters[user_id] = 1
        user_history[user_id] = deque(maxlen=500)
    else:
        user_counters[user_id] += 1    
    global model
    global url
    global user_key
    if(user_counters[user_id]<100 and user_counters[user_id]>0):

        prompt = message
        total_length = sum(len(chat) for chat in user_history.get(user_id, []) if isinstance(chat, str))
        print(total_length + len(prompt))


        # Remove oldest elements if total length exceeds the limit
        while total_length + len(prompt) > 5000: 
            removed_chat = user_history[user_id].popleft()
            total_length -= len(removed_chat)

        #message = ''.join(user_history[user_id].append(prompt)) 
        user_history[user_id].append(prompt) 
        message = ''.join(user_history[user_id])

        print(message) #testing
        if(user_id in user_key and user_id in url): #check if registerd endpoint
            res = a.chat(str(message), model,user_key[user_id],url[user_id])
            user_history[user_id].append("\n")
            user_history[user_id].append(res)
            user_history[user_id].append("\n")
            print(user_history[user_id])
            
            if(len(res)>MAX_MESSAGE_LENGTH):
                while len(res) > MAX_MESSAGE_LENGTH:
                    chunk = res[:MAX_MESSAGE_LENGTH]  # Get a chunk of the message within the limit
                    res = res[MAX_MESSAGE_LENGTH:]  # Remove the chunk from the original message

                

                    # Send the chunk as a separate message
                    await ctx.send(chunk)


            else:
                await ctx.send(res)
        else:
            await ctx.send("Error you didnt set Endpoint or Key")
        

    else:
        await ctx.send("You Reached Max Quota")



@bot.command()
async def ask(ctx, *, message):
    if(not await show_disclaimer(ctx)):
        return
    res = a.chatfree(str(pers) + "\nUser: " +str(message))['choices'][0]['message']['content']
    if(len(res)>MAX_MESSAGE_LENGTH):
        while len(res) > MAX_MESSAGE_LENGTH:
            chunk = res[:MAX_MESSAGE_LENGTH]  # Get a chunk of the message within the limit
            res = res[MAX_MESSAGE_LENGTH:]  # Remove the chunk from the original message
           
            # Send the chunk as a separate message
            await ctx.send(chunk)
    else:
        await ctx.send(res)




@bot.command()
async def info(ctx):
    await ctx.send(help)



@bot.command()
async def clear(ctx):
    global user_history
    user_history[ctx.user.id]=deque(maxlen=500)



@bot.command()
async def counter(ctx):
    user_id = ctx.author.id
    if user_id in user_counters:
        count = user_counters[user_id]
        await ctx.send(f"You have used the chat  {count} times.")
    else: 
        await ctx.send("You have not used chat yet.")



@bot.command()
async def query(ctx, *, question: str):
    if(not await show_disclaimer(ctx)):
        return
    # Check if there are any attachments in the message
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.size <= 5 * 1024 * 1024:
            
            # If there is an attachment, download it
            if attachment.filename.endswith('.txt'):
                # If it is a text file, download it
                #with open(attachment.filename, 'wb') as file:
                    #await attachment.save(file)

                # Now you can read the contents of the file
                with open(attachment.filename, 'r') as f:
                    content = f.read()
                    # Now you have the content of the file in the 'content' variable
                    await send_large_message(ctx,semantic_search(content, question))

            elif attachment.filename.endswith('.pdf'):
                file = await attachment.save()

                # Now you can read the contents of the file, for example:
                with open(file, 'rb') as f:
                    pdf_reader = PyPDF2.PdfFileReader(f)
                    content = ''
                    for page_num in range(pdf_reader.getNumPages()):
                        page = pdf_reader.getPage(page_num)
                        content += page.extractText()
                        await ctx.send(semantic_search(content, question))
            else:
                await ctx.send('Please upload a .txt or .pdf file.')
                return
        else:
            await ctx.send('Please upload a file that is less than 5 MB.')
            return           
    else:
        await ctx.send('Please upload a file')
        return

#AI removed for space issues
#model_name = 'sentence-transformers/all-MiniLM-L6-v2'
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)

async def send_large_message(ctx, message):
    while len(message) > MAX_MESSAGE_LENGTH:
        chunk = message[:MAX_MESSAGE_LENGTH]  # Get a chunk of the message within the limit
        message = message[MAX_MESSAGE_LENGTH:]  # Remove the chunk from the original message
        await ctx.send(chunk)  # Send the chunk as a separate message
    if message:  # Send any remaining part of the message
        await ctx.send(message)


def chunk_text(text, chunk_size):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embeddings(chunks):
    return np.array([text_to_embedding(chunk) for chunk in chunks])

def search(query, index, k=2):
    query_vector = text_to_embedding(query)
    D, I = index.search(np.array([query_vector]), k)
    return D, I

def semantic_search(text, query, chunk_size=300):
    # Split the text into chunks
    chunks = chunk_text(text, chunk_size)
    print(chunks[0])
    # Get embeddings for each chunk
    embeddings = get_embeddings(chunks)
    print(embeddings[0])
    # Build the FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Search the index
    D, I = search(query, index)

    # Concatenate the closest chunks
    result = 'Chunk 1:\n' + chunks[I[0][0]] + '\n\nChunk 2:\n' + chunks[I[0][1]]
    print(len(result))
    print(result)
    return result

def text_to_embedding(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Average the token-level embeddings to get a sentence-level embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().numpy()




bot.run("")





















