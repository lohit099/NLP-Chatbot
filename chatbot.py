import streamlit as st
import os
import json
import random
import datetime
import csv
import nltk
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Prepare data
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Custom styling
st.markdown("""
    <style>
        /* Sidebar Styling */
        .css-1d391kg {background-color: grey !important; color: black !important;}
        .css-1d391kg:hover {background-color: blue !important; color: black !important;}

        /* Deploy Button */
        .stButton > button {background-color: navy !important; color: black !important;}
        .stButton > button:hover {background-color: lightblue !important; color: white !important;}

        /* Main Section Styling */
        .stTextInput, .stTextArea, .stMarkdown {background-color: grey !important; color: black !important;}
        .stTextInput:hover, .stTextArea:hover, .stMarkdown:hover {background-color: lightblue !important; color: white !important;}

        /* Right Section (Main Content) Background */
        .stApp {background-color: lightblue !important;}
    </style>
""", unsafe_allow_html=True)

# Main UI
st.title("🤖 AI Chatbot: Your Smart Assistant")
menu = ["🏚️ Home", "⌛ Conversation History", "📝 About"]
choice = st.sidebar.selectbox("Menu", menu)

def save_conversation(user_input, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

if choice == "🏚️ Home":
    st.write("Start a conversation with the AI!")
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100)
        save_conversation(user_input, response)

elif choice == "⌛ Conversation History":
    st.header("Past Conversations")
    with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            st.text(f"User: {row[0]}")
            st.text(f"Chatbot: {row[1]}")
            st.text(f"Timestamp: {row[2]}")
            st.markdown("---")

elif choice == "📝 About":
    st.write("An AI-powered chatbot using NLP and Machine Learning for smart interactions. Future updates will include deep learning models for even better conversations!")

