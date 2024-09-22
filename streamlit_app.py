import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Check and download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    stop_words = set(stopwords.words('english'))

    for i in y:
        if i not in stop_words and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.title("Spam Email/SMS Detection")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
