from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth
import openai as OpenAI
import sqlite3
import gpt_chat

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



app = Flask(__name__) 
auth = HTTPTokenAuth(scheme='Bearer')
a =GPTChat()
shared_data = SharedData()
CORS(app)
def count_tokens(text):
    return len(text.split())

def chat(url, key, mod, messg):
    client = OpenAI(base_url=url, api_key=key)
    response = client.chat.completions.create(
    model=mod,
    messages=[{'role': 'user', 'content': messg}]
    )
    return response 

def verify_token(token):
    # Connect to the database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()

    # Check if token is in api_keys
    c.execute("SELECT user_id FROM shared_data WHERE api_key=?", (token,))
    result = c.fetchone()

    conn.close()

    if result is not None:
        return result[0]
    return None

@app.route("/v1/chat/completions", methods=['GET','POST'])
@auth.login_required
def home():
    user_id = auth.current_user()
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT model, url FROM shared_data WHERE user_id=?", (user_id,))
    result = c.fetchone()

    conn.close()

    if result is None:
        return jsonify({"error": "No data found for the user."}), 400

    model, url = result

    data = request.get_json()
    user_message = data['messages'][-1]['content']

    if model == 'Gemini':
        response = a.geminiChat(self, user_message)
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": count_tokens(user_message),
                "completion_tokens": count_tokens(response),
                "total_tokens": count_tokens(user_message) + count_tokens(response)
            }
        }
    else:
        response = chat(url, token, model, user_message)

    return jsonify(response)


app.run()
