import pandas as pd
import re

# Read JD Excel
df = pd.read_excel("jd_data.xlsx")

# Select first JD
jd = str(df.loc[0, "Job_Desc"]).lower()

# Skill Dictionary
skills = [
    "python",
    "sql",
    "java",
    "aws",
    "azure",
    "snowflake",
    "talend",
    "informatica",
    "databricks",
    "spark",
    "pyspark",
    "airflow",
    "kafka",
    "docker",
    "kubernetes",
    "power bi",
    "tableau",
    "excel",
    "etl",
    "machine learning",
    "deep learning",
    "tensorflow",
    "pandas",
    "numpy"
]

matched_skills = []

for skill in skills:
    if re.search(r"\b" + re.escape(skill) + r"\b", jd):
        matched_skills.append(skill)

print("Skills found in JD:\n")

for skill in matched_skills:
    print("✔", skill)