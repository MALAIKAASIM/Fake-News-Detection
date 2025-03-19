import string
import streamlit as st 
import pickle
import re
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load trained model and vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file) 

# Load trained model and vectorizer
with open("rf_model.pkl", "rb") as model_file:
   model = pickle.load(model_file)

    
# Initialize Stemming and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load English stopwords
stop_words = set(stopwords.words('english'))

def word_drop(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove text inside square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize the text into words
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Apply Stemming
    words = [stemmer.stem(word) for word in words]

    # Apply Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join words back into a string
    return " ".join(words)

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a News Article below to check whether it is Fake or Real. ")

# User input
input_text = st.text_area("News Article:", "")

if st.button("Check News"):
    if input_text.strip():
        cleaned_text = word_drop(input_text)  # Preprocess input
        transform_input = vectorizer.transform([cleaned_text])  # Convert to numerical format
        
        # Ensure feature size matches
        if transform_input.shape[1] != model.n_features_in_:
            st.error(f"Feature mismatch! Model expects {model.n_features_in_} features, but got {transform_input.shape[1]}.")
        else:
            prediction = model.predict(transform_input)

            if prediction[0] == 1:
                st.success("The News is Real!")
            else:
                st.error("The News is Fake!")
    else:
        st.warning("Please enter some text to analyze.")