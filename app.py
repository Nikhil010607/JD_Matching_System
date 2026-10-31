
import fitz
import io
import html
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.skill_extractor import extract_skills


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="TalentMatch | Recruitment Intelligence",
    page_icon="assets/hexaware_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- PROFESSIONAL CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F6FA;
}

/* Remove unnecessary top space */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #102F4F;
    border-right: 1px solid #244B6B;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

.sidebar-heading {
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    color: #B8CCDE;
    font-size: 12px;
    margin-bottom: 25px;
}

.sidebar-info {
    background: #19466D;
    padding: 18px;
    border-radius: 12px;
    margin-top: 20px;
    line-height: 2;
    font-size: 13px;
}

/* Main header */
.main-header {
    background: linear-gradient(115deg, #123A60, #176BB0);
    padding: 34px 40px;
    border-radius: 16px;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(16, 47, 79, 0.13);
}

.header-label {
    color: #B9D8F4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.main-header h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin: 12px 0;
}

.main-header p {
    color: #E0EDF8;
    font-size: 14px;
    margin: 0;
}

/* Section headings */
.section-heading {
    font-size: 20px;
    font-weight: 700;
    color: #183B5B;
    margin-top: 25px;
    margin-bottom: 14px;
}

.section-description {
    color: #718096;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 20px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E5EAF0;
    padding: 20px;
    border-radius: 13px;
    box-shadow: 0 3px 12px rgba(20, 40, 70, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #123A60;
    font-size: 26px;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: #176BB0;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 22px;
    min-height: 45px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #104B7C;
    color: white;
}

/* Download button */
.stDownloadButton > button {
    background: #123A60;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 44px;
    font-weight: 600;
}

.stDownloadButton > button:hover {
    background: #176BB0;
    color: white;
}

/* Input elements */
div[data-baseweb="select"] > div {
    border-radius: 9px;
    border-color: #DCE3EC;
    background: white;
}

section[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #A8BBD0;
    border-radius: 12px;
    padding: 15px;
}

/* Candidate cards */
.candidate-card {
    background: white;
    border: 1px solid #E3E9F0;
    border-radius: 13px;
    padding: 23px;
    margin: 15px 0;
    box-shadow: 0 3px 12px rgba(20, 40, 70, 0.04);
}

.candidate-rank {
    color: #176BB0;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.candidate-name {
    color: #183B5B;
    font-size: 18px;
    font-weight: 700;
    margin: 10px 0;
}

.score-number {
    color: #176BB0;
    font-size: 27px;
    font-weight: 800;
}

.skill-tag {
    display: inline-block;
    background: #EAF3FC;
    color: #15558A;
    padding: 6px 11px;
    border-radius: 6px;
    margin: 4px 5px 4px 0;
    font-size: 12px;
    font-weight: 600;
}

.missing-tag {
    display: inline-block;
    background: #FFF0EF;
    color: #B54747;
    padding: 6px 11px;
    border-radius: 6px;
    margin: 4px 5px 4px 0;
    font-size: 12px;
    font-weight: 600;
}

/* Divider */
hr {
    border-color: #E1E7EF;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.image("assets/hexaware_logo.png", width=175)

    st.markdown(
        '<div class="sidebar-heading">Recruitment Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">AI-powered talent evaluation platform</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("**WORKSPACE**")
    st.markdown("Candidate Screening")
    st.markdown("Job Description Matching")
    st.markdown("Recruiter Analytics")
    st.markdown("Report Generation")

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-info">
        <b>PROJECT INFORMATION</b><br>
        <b>Developer</b><br>
        Nikhil Nathan<br>
        <b>Organization</b><br>
        Hexaware Technologies<br>
        <b>Technology</b><br>
        Python | Machine Learning<br>
        NLP | Streamlit
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("AI Resume Screening System")



# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="main-header">
<div class="header-label">HEXAWARE TECHNOLOGIES</div>
<h1>Talent Acquisition Dashboard</h1>
<p>
Intelligent Resume Screening and Job Description Matching
powered by Machine Learning and Natural Language Processing.
</p>
</div>
""", unsafe_allow_html=True)



# ---------------- LOAD JOB DESCRIPTIONS ----------------

try:
    df = pd.read_excel("jd_data.xlsx")
    if "Job_Desc" not in df.columns or df.empty:
        st.error("The Excel file must contain a non-empty Job_Desc column.")
        st.stop()
except Exception as e:
    st.error(f"Unable to load Job Description data: {e}")
    st.stop()


# ---------------- JOB DESCRIPTION SECTION ----------------

st.markdown(
    '<div class="section-heading">Job Requirements</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Select the role for which candidates need to be evaluated.</div>',
    unsafe_allow_html=True
)

selected_jd = st.selectbox(
    "Select Job Description",
    df["Job_Desc"].dropna().tolist()
)

with st.expander("View complete job description"):
    st.write(selected_jd)


# ---------------- RESUME UPLOAD ----------------

st.markdown(
    '<div class="section-heading">Candidate Resume Upload</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">Upload multiple PDF resumes to evaluate candidate suitability.</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload candidate resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload PDF resumes for candidate screening."
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} candidate resume(s) uploaded.")

    with st.expander("View uploaded files"):
        for file in uploaded_files:
            st.write(file.name)


# ---------------- CANDIDATE ANALYSIS ----------------

if st.button("Analyze Candidates", use_container_width=True):

    if not uploaded_files:
        st.warning("Please upload at least one resume.")
        st.stop()

    results = []

    jd_skills = extract_skills(selected_jd)

    with st.spinner("Analyzing resumes and calculating matching scores..."):

        for file in uploaded_files:

            try:
                doc = fitz.open(
                    stream=file.getvalue(),
                    filetype="pdf"
                )

                resume_text = ""

                for page in doc:
                    resume_text += page.get_text()

                doc.close()

                if not resume_text.strip():
                    st.warning(f"No readable text found in {file.name}.")
                    continue

                # TF-IDF similarity
                documents = [selected_jd, resume_text]

                try:
                    vectorizer = TfidfVectorizer(
                        stop_words="english"
                    )

                    vectors = vectorizer.fit_transform(documents)

                    similarity = cosine_similarity(vectors)[0][1]

                    tfidf_score = round(similarity * 100, 2)

                except ValueError:
                    tfidf_score = 0

                # Extract resume skills
                resume_skills = extract_skills(resume_text)

                matched_skills = sorted(
                    set(jd_skills) & set(resume_skills)
                )

                missing_skills = sorted(
                    set(jd_skills) - set(resume_skills)
                )

                # Skill score
                if len(jd_skills) == 0:
                    skill_score = 0
                else:
                    skill_score = (
                        len(matched_skills) / len(jd_skills)
                    ) * 100

                # Weighted final score
                overall_score = (
                    0.6 * skill_score
                ) + (
                    0.4 * tfidf_score
                )

                results.append({
                    "Resume": file.name,
                    "Match Score": round(overall_score, 2),
                    "TF-IDF Score": tfidf_score,
                    "Skill Score": round(skill_score, 2),
                    "Matched Skills": matched_skills,
                    "Missing Skills": missing_skills
                })

            except Exception as e:
                st.warning(f"Could not process {file.name}: {e}")

    if not results:
        st.error("No resumes could be successfully analyzed.")
        st.stop()

    results = sorted(
        results,
        key=lambda x: x["Match Score"],
        reverse=True
    )

    st.session_state["candidate_results"] = results


# ---------------- DISPLAY RESULTS ----------------

if "candidate_results" in st.session_state:

    results = st.session_state["candidate_results"]

    best_candidate = results[0]

    average_score = sum(
        c["Match Score"] for c in results
    ) / len(results)

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Recruitment Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Summary of the candidate screening results.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Candidates",
            len(results)
        )

    with col2:
        st.metric(
            "Highest Match",
            f"{best_candidate['Match Score']:.2f}%"
        )

    with col3:
        st.metric(
            "Average Match",
            f"{average_score:.2f}%"
        )

    with col4:
        st.metric(
            "Strong Matches",
            sum(1 for c in results if c["Match Score"] >= 80)
        )

    # ---------------- BEST CANDIDATE ----------------

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Top Recommended Candidate</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="candidate-card">

        <div class="candidate-rank">Highest Ranked Candidate</div>

        <div class="candidate-name">
            {html.escape(best_candidate['Resume'])}
        </div>

        <div class="score-number">
            {best_candidate['Match Score']:.2f}%
        </div>

        <div style="color:#718096;font-size:13px;margin-top:5px;">
            Overall Matching Score
        </div>

    </div>
    """, unsafe_allow_html=True)

    # ---------------- EXCEL REPORT ----------------

    report_data = []

    for rank, candidate in enumerate(results, start=1):

        report_data.append({
            "Rank": rank,
            "Resume": candidate["Resume"],
            "Match Score (%)": candidate["Match Score"],
            "Skill Score (%)": candidate["Skill Score"],
            "TF-IDF Score (%)": candidate["TF-IDF Score"],
            "Matched Skills": ", ".join(candidate["Matched Skills"]),
            "Missing Skills": ", ".join(candidate["Missing Skills"])
        })

    report_df = pd.DataFrame(report_data)

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        report_df.to_excel(
            writer,
            index=False,
            sheet_name="Recruiter Report"
        )

    st.download_button(
        label="Download Recruitment Report",
        data=excel_buffer.getvalue(),
        file_name="Recruitment_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ---------------- CANDIDATE RANKINGS ----------------

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Candidate Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Ranked candidates with detailed skill matching analysis.</div>',
        unsafe_allow_html=True
    )

    for i, candidate in enumerate(results, start=1):

        with st.container():

            st.markdown(f"""
            <div class="candidate-card">

                <div class="candidate-rank">
                    RANK {i}
                </div>

                <div class="candidate-name">
                    {html.escape(candidate['Resume'])}
                </div>

                <div class="score-number">
                    {candidate['Match Score']:.2f}%
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.progress(
                min(candidate["Match Score"] / 100, 1.0)
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("**Matched Skills**")

                if candidate["Matched Skills"]:

                    tags = "".join(
                        f'<span class="skill-tag">{html.escape(skill)}</span>'
                        for skill in candidate["Matched Skills"]
                    )

                    st.markdown(tags, unsafe_allow_html=True)

                else:
                    st.caption("No matching skills identified.")

            with col2:

                st.markdown("**Skills to Improve**")

                if candidate["Missing Skills"]:

                    tags = "".join(
                        f'<span class="missing-tag">{html.escape(skill)}</span>'
                        for skill in candidate["Missing Skills"]
                    )

                    st.markdown(tags, unsafe_allow_html=True)

                else:
                    st.caption("All identified JD skills matched.")

            st.markdown("")

    st.markdown("---")

    st.caption(
        "Talent Acquisition Dashboard | "
        "Machine Learning Based Resume Screening"
    )
