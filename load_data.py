import pandas as pd

df = pd.read_csv("data/spam.csv", encoding="latin-1")

print(df.head())

print(df.columns)

print(df.shape)