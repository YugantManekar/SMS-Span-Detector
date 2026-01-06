import streamlit as st
import pickle

st.title("SMS Spam Classifier")

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

msg = st.text_input("Enter message")

if st.button("Predict"):
    vector = tfidf.transform([msg])
    result = model.predict(vector)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
