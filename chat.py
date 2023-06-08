import streamlit as st

import pandas as pd
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import streamlit.web.bootstrap


st.set_page_config(page_title="ChatMATE")

initial_responses = [
    "Hello there! I'm ChatMATE, your personal AI assistant. How can I assist you today?",
    "Greetings! I'm here to lend a helping hand. What can I do for you?",
    "Welcome! I'm ChatMATE, ready to assist you. How may I be of service?",
    "Hi, I'm your friendly AI assistant, ChatMATE. How can I assist you today?",
    "Good day! I'm ChatMATE, your AI companion. How may I assist you today?",
    "Hello! I'm here to help. What can I assist you with today?",
    "Greetings! I'm ChatMATE, your AI assistant at your service. How can I assist you?",
    "Hi there! I'm your trusty AI assistant, ChatMATE. How may I help you today?",
    "Welcome! I'm ChatMATE, the AI assistant designed to assist you. What can I do for you?",
    "Hello! I'm here to assist you. How may I be of help today?"
]

# Generate empty lists for bot_response and user_input.
## bot_response stores AI generated responses
if 'bot_response' not in st.session_state:
    st.session_state['bot_response'] = [random.choice(initial_responses)]

## user_input stores User's questions
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ['Hi!']

st.title("ChatMATE -Your AI Assistant")
st.markdown(
    """
    <style>
    .css-1qj5x5v {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)



logo_image = Image.open(r'')

# Resize the logo image to a smaller size
resized_logo = logo_image.copy()
resized_logo.thumbnail((200, 200))

# Display the resized logo image using Streamlit
st.image(resized_logo, width=50, use_column_width='always')

# Apply CSS styling to center align the image
st.markdown(
    """
    <style>
    .stImage > img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()



# User input
## Function for taking user provided prompt as input
def get_input():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_input()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def preprocess_text(text):
    # Implement your own text preprocessing logic here
    return text


def generate_response(prompt):
    df = pd.read_csv(r"")

    # Replace NaN values with empty strings
    df.fillna('', inplace=True)

    df1 = df[['Market Name', 'Sub-Stream', 'SW Release', 'Title', 'Use Case / Problem statement',
              'Use Case/Issue Description ']].copy()
    df1['Combined'] = df1.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    df2 = df[['Work Around']].copy()

    # Reset the indices to match between df1 and df2
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(df1['Combined'])

    # Preprocess the user prompt
    preprocessed_prompt = preprocess_text(prompt)

    keywords = [word.lower() for word in word_tokenize(preprocessed_prompt) if word.lower() not in stop_words]
    keyword_matrix = vectorizer.transform([' '.join(keywords)])
    similarity_scores = cosine_similarity(tfidf_matrix, keyword_matrix)

    match_indices = similarity_scores.argmax(axis=0)  # Find the indices with maximum similarity score
    for index in match_indices:
        if similarity_scores[index] > 0:
            break  # Exit the loop after finding the first match

    if similarity_scores[index] > 0:
        answer_main = df2.loc[index, 'Work Around']

        chatbot = hugchat.ChatBot(cookie_path=r"C:\chatbot req\cookies.json")
        answer = f"I encountered an issue with '{df1.loc[index, 'Combined']}' and the recommended solution for this problem is as follows: '{answer_main}'. Could you please provide guidance on how to effectively address this issue using the appropriate technical terms?"
        response = chatbot.chat(answer)
        return response
    else:
        return "I'm sorry, I don't have an answer to that question."

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.user_input.append(user_input)
        st.session_state.bot_response.append(response)

    if st.session_state['bot_response']:
        for i in range(len(st.session_state['bot_response'])):
            message(st.session_state['user_input'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['bot_response'][i], key=str(i))

