import fitz
import os

resume_folder = "resumes"

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(resume_folder, file)

        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        print("\n")
        print("=" * 50)
        print(file)
        print("=" * 50)

        print(text[:1000])