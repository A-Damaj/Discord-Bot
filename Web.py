import streamlit as st
from PIL import Image
import gpt_chat
from gpt_chat import GPTChat

def get_gpt_response(message):
    if model=="gemini" and not endpoint and not key:
        return a.geminiChat(message)
    elif model=="gpt-3" and not endpoint and not key:
        return a.chatfree(message)

    if not model or not endpoint or not key:
        return "Model, Endpoint, or Key is missing. Please enter these values in the sidebar."

    return a.chat(message,model,key,endpoint)

# Initialize session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
a=GPTChat()
st.sidebar.subheader("Model Configuration")
model = st.sidebar.text_input("Enter the Model")
endpoint = st.sidebar.text_input("Enter the Endpoint")
key = st.sidebar.text_input("Enter the Key")
user_message = st.chat_input("Your Message")

# If the user has sent a message
if user_message:
    # Append the user message to the chat history
    st.session_state.chat_history.append(("user", user_message))

    # Get the GPT response
    gpt_response = get_gpt_response(user_message)
    st.session_state.chat_history.append(("assistant", gpt_response))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Display the image
    st.image(img, caption="Uploaded Image")
    image_question = st.text_input("Your question about the image")
    if image_question:
        st.session_state.chat_history.append(("user", image_question))
        gpt_image_response = a.gemimg(image_question, img)
        st.session_state.chat_history.append(("assistant", gpt_image_response))

# Display the chat history
for author, message in st.session_state.chat_history[-6:]:
    with st.chat_message(author):
        st.write(message)



