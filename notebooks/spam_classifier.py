import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]

df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham' : 0, 'spam' : 1})
print("\nEncoded Labels:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSpam vs Ham Count:")
print(df['label'].value_counts())

X= df['message']
y= df['label']

print("\nFeatures:")
print(X.head())
print("\nlabels:")
print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)
print("\nTraining samples", len(X_train))
print("Training samples:", len(X_test))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf= vectorizer.transform(X_test)
print("\nTraining matrix Shape:")
print(X_train_tfidf.shape)
print("\nTesting matrix Shape:")
print(X_test_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("\nModel trained successfully.")

from sklearn.metrics import accuracy_score
y_pred= model.predict(X_test_tfidf)
accuracy= accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

