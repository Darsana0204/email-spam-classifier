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
print("\nYraining samples", len(X_train))
print("Training samples:", len(X_test))