import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the 'punkt' and 'stopwords' resources if not already available
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess the text input
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize

    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters

    # Filter stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load pre-trained models
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Streamlit app title
st.title("Spam Email/SMS Detection")

# Text input field for users
input_sms = st.text_area("Enter the message")

# When user clicks 'Predict'
if st.button('Predict'):
    if input_sms.strip() != "":
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # Vectorize the input
        try:
            vector_input = tfidf.transform([transformed_sms])
        except Exception as e:
            st.error(f"Error vectorizing input: {e}")
            st.stop()

        # Predict using the loaded model
        try:
            result = model.predict(vector_input)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a valid message.")
