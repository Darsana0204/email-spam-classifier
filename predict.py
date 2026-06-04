import pickle

with open("model.pkl", "rb")  as f:
    model= pickle.load(f)

with open("vectorizer.pkl","rb") as f:
    vectorizer= pickle.load(f)

def predict_spam(message):
    text= vectorizer.transform([message])
    result= model.predict(text)
    return "Spam" if result[0]== 1 else "Ham"

    