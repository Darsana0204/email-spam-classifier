import pandas as pd

df= pd.read_csv("data/spam.csv", encoding="latin-1")
df=df[['v1','v2']]
df.columns= ['label','message']
df['label']=df['label'].map({'ham':0, 'spam':1})
print(df.head())

from sklearn.model_selection import train_test_split
train_df, test_df= train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print("Training sample:", len(train_df))
print("Testing samples:", len(test_df))

from transformers import AutoTokenizer
tokenizer= AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded successfully!")