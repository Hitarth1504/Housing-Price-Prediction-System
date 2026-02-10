import pandas as pd

df = pd.read_csv("housing.csv")
print(df.isnull().sum())

dr = df.dropna()
print(dr.isnull().sum())
dr.to_csv("cleandata.csv")