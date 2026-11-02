import pandas as pd

# Dataset load karna
df = pd.read_csv("data/student-mat.csv", sep=";")

# Pehli 5 rows dekhna
print(df.head())

# Dataset ka size
print("\nDataset shape:", df.shape)

# Saare columns
print("\nColumns:")
print(df.columns.tolist())

# Missing values check karna
print("\nMissing values:")
print(df.isnull().sum())

# Final grade ka average
print("\nAverage final grade:", df["G3"].mean())