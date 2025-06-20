import streamlit as st
import pickle
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Set nltk data path
nltk.data.path.append("C:/Users/DELL/AppData/Roaming/nltk_data")

# Load model (pipeline with vectorizer + classifier)
model = pickle.load(open(r"C:\Users\JAHNAVI\Downloads\Project1\Project1\model.pkl", 'rb'))


# Text transformation
def transform_text(text):
  text = text.lower()
  words = text.split()
  cleaned = [
    word for word in words
    if word not in string.punctuation and word not in stopwords.words('english')
  ]
  return " ".join(cleaned)


# Streamlit UI
st.title("Email Spam Detector")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
  if input_sms:
    transformed_sms = transform_text(input_sms)
    result = model.predict([transformed_sms])[0]

    if result == 1:
      st.header("Spam ❌")
    else:
      st.header("Not Spam ✅")




