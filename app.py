import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the saved vectorizer and model using pickle
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the working directory.")

# Streamlit app layout
st.title("Spam Email/SMS Detection")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess the input message
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])
        # 3. Make the prediction
        result = model.predict(vector_input)[0]
        # 4. Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message to analyze.")
