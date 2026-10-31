import pandas as pd
import fitz
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- READ JD ----------------

df = pd.read_excel("jd_data.xlsx")

# Select first JD for testing
jd_text = str(df.loc[0, "Job_Desc"])

print("\nSelected JD:\n")
print(jd_text[:500])

# ---------------- READ RESUMES ----------------

resume_folder = "resumes"

resume_names = []
resume_texts = []

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(resume_folder, file)

        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        resume_names.append(file)
        resume_texts.append(text)

# ---------------- TF-IDF ----------------

documents = [jd_text] + resume_texts

vectorizer = TfidfVectorizer(stop_words="english")

vectors = vectorizer.fit_transform(documents)

# ---------------- COSINE SIMILARITY ----------------

scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

results = []

for name, score in zip(resume_names, scores):

    results.append((name, score * 100))

results.sort(key=lambda x: x[1], reverse=True)

print("\nTop Matching Resumes\n")

for i, (name, score) in enumerate(results, start=1):

    print(f"{i}. {name}")
    print(f"Match Score : {score:.2f}%\n")