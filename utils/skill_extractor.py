import re

# Master Skill List
SKILLS = [
    "python", "java", "c++", "sql", "mysql", "oracle", "postgresql",
    "mongodb", "aws", "azure", "gcp", "snowflake", "databricks",
    "spark", "pyspark", "hadoop", "hive", "kafka", "airflow",
    "talend", "informatica", "ssis", "etl",
    "docker", "kubernetes", "linux", "git", "github",
    "power bi", "tableau", "excel",
    "machine learning", "deep learning",
    "tensorflow", "keras", "pandas", "numpy", "scikit-learn"
]

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        if re.search(r"\b" + re.escape(skill) + r"\b", text):

            found_skills.append(skill)

    return sorted(list(set(found_skills)))