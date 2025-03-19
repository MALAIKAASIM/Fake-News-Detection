import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
with open("rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    """Preprocess input text: remove URLs, punctuation, stopwords, and apply stemming & lemmatization."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    return ' '.join(words)

def predict_news(text):
    """Predict if the news is real or fake."""
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    return "Real News" if prediction == 1 else "Fake News"

# Streamlit UI
st.title("Fake News Detection App")
st.write("Enter a news article below to check if it's real or fake.")

# Text Input
news_input = st.text_area("Paste your news article here:", "")

if st.button("Analyze News"):
    if news_input.strip() == "":
        st.warning(" Please enter a news article.")
    else:
        result = predict_news(news_input)
        if result == "Real News":
            st.success(" This news article appears to be Real.")
        else:
            st.error(" This news article appears to be Fake.")

st.write("Developed by [Your Name]")
