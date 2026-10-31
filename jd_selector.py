import pandas as pd

# Read Excel
df = pd.read_excel("jd_data.xlsx")

print("\nTotal JDs :", len(df))

print("\nAvailable Job Descriptions:\n")

for i in range(10):
    print(f"{i} -> {df.loc[i, 'Job_Desc'][:80]}...")