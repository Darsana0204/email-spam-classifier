import pickle

with open("model.pkl","rb") as file:
    model =pickle.load(file)
with open("vectorizer.pkl","rb") as file:
    vectorizer =pickle.load(file)

message="Congratulations! You have won a free iPhone. claim now!"
message_tfidf= vectorizer.transform([message])
prediction= model.predict(message_tfidf)
if prediction[0]==1:
    print("Spam")
else:
    print("Ham")