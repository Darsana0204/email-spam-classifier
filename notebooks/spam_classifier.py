import pandas as pd

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

print(df.columns)

df = df[['v1', 'v2']]

df.columns = ['label', 'message']

print(df.head())