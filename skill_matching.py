import pandas as pd
import fitz
import os

from utils.skill_extractor import extract_skills

# ---------------- READ JD ----------------

df = pd.read_excel("jd_data.xlsx")

jd_text = str(df.loc[0, "Job_Desc"])

jd_skills = extract_skills(jd_text)

print("\nJD Skills:")
print(jd_skills)

# ---------------- READ RESUMES ----------------

resume_folder = "resumes"

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(resume_folder, file)

        doc = fitz.open(pdf_path)

        resume_text = ""

        for page in doc:
            resume_text += page.get_text()

        resume_skills = extract_skills(resume_text)

        matched = list(set(jd_skills) & set(resume_skills))
        missing = list(set(jd_skills) - set(resume_skills))

        print("\n======================================")
        print("Resume :", file)

        print("Matched Skills :", matched)
        print("Missing Skills :", missing)