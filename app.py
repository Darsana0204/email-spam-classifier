import streamlit as st
import pickle

with open("model.pkl", "rb") as file:
    model= pickle.load(file)

with open("vectorizer.pkl","rb") as file:
    vectorizer= pickle.load(file)

st.title("Email Spm Classifier")
message= st.text_area("Enter your message")
if st.button("Predict"):
    transformed_message= vectorizer.transform([message])
    prediction= model.predict(transformed_message)
    if prediction[0]==1:
        st.error("Spam Email")
    else:
        st.success("Ham Email")