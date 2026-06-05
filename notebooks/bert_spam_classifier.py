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

train_encodings =tokenizer(train_df["message"].tolist(), truncation=True, padding=True)
test_encodings= tokenizer(test_df["message"].tolist(),truncation=True, padding=True)
print("Training messages tokenized!")
print("Testing messages tokenized!")

import torch

class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings= encodings
        self.labels= labels.tolist()

    def __getitem__(self, idx):
        item= {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"]= torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
train_dataset = SpamDataset(
    train_encodings, train_df["label"]
    )
test_dataset= SpamDataset(test_encodings, test_df["label"])
print("Training dataset created!")
print("Testing dataset created!")

print(len(train_dataset))
print(len(test_dataset))

from transformers import AutoModelForSequenceClassification
model= AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("DistilBERT model loaded successfully")