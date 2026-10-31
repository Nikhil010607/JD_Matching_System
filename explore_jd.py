import pandas as pd

df = pd.read_excel("jd_data.xlsx")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())