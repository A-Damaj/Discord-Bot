## Discord Bot Project: A Comprehensive Guide

This repository contains the code for a versatile Discord bot designed to interact with various AI models, including Chat-GPT and Gemini. The bot offers a range of functionalities, from basic chat interactions to image analysis and semantic search.

### Key Features:

* **Multi-Model Support:** The bot can interact with different AI models like Bard, Gemini, and potentially others through API integrations. 
* **Chat Interaction:** Engage in conversations with the bot, leveraging the capabilities of the chosen AI model. 
* **Image Analysis:** Ask questions about uploaded images and receive insights generated by the AI.
* **Semantic Search:** Upload text files or PDFs and ask questions to receive relevant excerpts based on semantic similarity.
* **User Management:** The bot includes features like user quotas and chat history management.
* **Proxy Integration:** Set up a proxy server using ngrok to enable external access to the bot. (Demo purposes only)

### Getting Started:

1. **Prerequisites:**
    * Python 3.x with required libraries (Flask, discord.py, openai, etc.)
    * Discord bot token
    * API keys for the desired AI models (Bard, Gemini, etc.)
    * ngrok (optional, for proxy setup)
2. **Installation:**
    * Clone the repository to your local machine.
    * Install the required Python libraries using `pip install -r requirements.txt`.
3. **Configuration:**
    * Update the bot token, API keys, and other settings in the respective files.
    * Set up the database according to the instructions in `setup_database()` functions.
4. **Running the Bot:**
    * Start the Flask server using `python api.py`.
    * Run the Discord bot using `python bot.py`.
    * Optionally, start the ngrok server using `ngrok http 5000`.

### Usage:

* **Chat Commands:**
    * `/chat <message>`: Start a conversation with the bot using the chosen AI model.
    * `/ask <message>`: Ask a question and receive a response from Bard (free version).
    * `/gemini_chat <message>`: Interact with Google's Gemini model directly.
    * `/gemini_image <message>`: Analyze an attached image using Gemini and provide insights. 
    * `/clear`: Clear your chat history with the bot.
    * `/counter`: Check how many times you've used the chat feature.
    * `/query <question>`: Upload a .txt or .pdf file and ask a question to retrieve relevant information.
* **Configuration Commands:**
    * `/gen_key`: Generate a unique key for API access (required for some features).
    * `/endpoint <url>`: Set the endpoint URL for the chosen AI model.
    * `/SetKey <api_key>`: Set the API key for the chosen AI model.
    * `/SetModel <model_name>`: Select the desired AI model (e.g., Bard, Gemini).
* **Proxy and Server Commands:**
    * `/run_ngrok`: Start the ngrok server for external access to the bot.
    * `/proxy`: Get the public URL provided by ngrok.
    * `/stop_ngrok`: Stop the ngrok server.
    * `/run_server`: Start the Flask server for API interactions.
    * `/stop_server`: Stop the Flask server.
* **Additional Commands:**
    * `/info`: Get information about the bot and its functionalities.


## Disclaimer

This bot and its components are designed for educational and experimental purposes. Please use responsibly and ensure compliance with all applicable laws and ethical guidelines when interacting with AI models.
