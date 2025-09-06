# app.py

import streamlit as st
import joblib
import re
import nltk   # <-- Added
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Download NLTK stopwords at runtime (needed on Streamlit Cloud)
# ----------------------------
nltk.download('stopwords')  # <-- Added

# ----------------------------
# 1. Load trained model & vectorizer
# ----------------------------
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ----------------------------
# 2. Initialize stemmer and stopwords
# ----------------------------
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# 3. Preprocessing function
# ----------------------------
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [port_stem.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# ----------------------------
# 4. Streamlit UI layout
# ----------------------------
st.set_page_config(page_title="Fake News Predictor", layout="wide")

# Custom styling for better look
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .stButton button {background-color: #ff4b4b; color: white; height: 3em; width: 100%; font-size:18px;}
    .stTextInput input {height: 2em; font-size:16px;}
    .stTextArea textarea {height: 200px; font-size:16px;}
    h1, h2 {text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Predictor")
st.write("Enter a news **Title** and **Text** below to predict whether it is REAL or FAKE.")

# Input fields
title = st.text_input("Title")
text = st.text_area("Text")

# Prediction button
if st.button("Predict"):
    if title.strip() != "" and text.strip() != "":
        # Combine and preprocess
        content = title + ' ' + text
        cleaned = clean_text(content)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]

        # Display result
        result = "REAL 🟢" if prediction == 0 else "FAKE 🔴"
        st.markdown(f"<h2>{result}</h2>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter both title and text!")