# app.py
import joblib
import streamlit as st
import PyPDF2
import docx

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Resume Analyzer")
st.write("Upload your resume file or paste text along with a job description.")

# File upload option
uploaded_file = st.file_uploader("Upload Resume File (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

resume_input = ""
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        resume_input = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_input = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        resume_input = " ".join([para.text for para in doc.paragraphs])

# Fallback: manual paste
resume_text_area = st.text_area("Or Paste Resume Text", value=resume_input)
job_desc_input = st.text_area("Paste Job Description")

# Expanded skill dictionary (100+ skills across domains)
skill_list = [
    # IT & Programming
    "python","java","c","c++","c#","javascript","typescript","html","css","react","angular","vue",
    "nodejs","django","flask","spring boot","pandas","numpy","sql","mongodb","postgresql","mysql",
    "git","docker","kubernetes","aws","azure","gcp","linux","bash","devops","machine learning",
    "deep learning","nlp","data analysis","data visualization","excel","power bi","tableau",
    "cloud computing","cybersecurity","networking",
    # Engineering
    "cad","autocad","solidworks","matlab","mechanical design","electrical circuits","structural analysis",
    "civil engineering","surveying","materials science",
    # Healthcare
    "nursing","patient care","pharmacology","clinical experience","empathy","medical procedures",
    "diagnosis","public health","anatomy","physiology",
    # Business & Management
    "project management","leadership","communication","teamwork","problem solving","strategic planning",
    "budgeting","finance","accounting","auditing","business analysis","marketing","sales","customer service",
    "negotiation","supply chain","operations management","entrepreneurship",
    # Creative & Design
    "graphic design","photoshop","illustrator","indesign","creativity","typography","ui design","ux design",
    "wireframing","branding","video editing","animation",
    # Education
    "teaching","curriculum design","lesson planning","classroom management","assessment","mentoring","training",
    # Soft Skills
    "adaptability","critical thinking","time management","collaboration","presentation","organization",
    "decision making","emotional intelligence"
]

# Simple domain tagging
domains = {
    "IT": ["python","java","c","c++","sql","aws","azure","git","docker","machine learning"],
    "Healthcare": ["nursing","patient care","pharmacology","clinical experience","anatomy","physiology"],
    "Engineering": ["cad","autocad","solidworks","structural analysis","civil engineering","matlab"],
    "Business": ["project management","finance","accounting","marketing","sales","leadership"],
    "Creative": ["graphic design","photoshop","illustrator","branding","animation","ui design","ux design"],
    "Education": ["teaching","curriculum design","lesson planning","classroom management"]
}

if st.button("Analyze Resume"):
    if resume_text_area.strip() and job_desc_input.strip():
        # Combine resume + job description for ML model
        combined_input = resume_text_area + " " + job_desc_input
        resume_vec = vectorizer.transform([combined_input])
        prediction = model.predict(resume_vec)[0]
        probability = model.predict_proba(resume_vec)[0]

        st.subheader("📊 Prediction Results")
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence: {max(probability)*100:.2f}%")

        # Extract skills
        job_skills = [skill for skill in skill_list if skill in job_desc_input.lower()]
        resume_skills = [skill for skill in skill_list if skill in resume_text_area.lower()]
        missing_skills = set(job_skills) - set(resume_skills)

        # Calculate match score
        if job_skills:
            match_score = (len(set(job_skills) & set(resume_skills)) / len(job_skills)) * 100
        else:
            match_score = 0

        # Domain detection
        resume_domain = None
        job_domain = None
        for domain, keywords in domains.items():
            if any(skill in resume_skills for skill in keywords):
                resume_domain = domain
            if any(skill in job_skills for skill in keywords):
                job_domain = domain

        # Override prediction if domains mismatch
        if resume_domain and job_domain and resume_domain != job_domain:
            prediction = "Not Suitable"
            st.write("⚠️ Domain mismatch detected: Resume domain is "
                     f"{resume_domain}, Job domain is {job_domain}.")

        st.subheader("✅ Matched Skills")
        st.write(", ".join(resume_skills) if resume_skills else "None")

        st.subheader("❌ Skills to Improve (Missing Skills)")
        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            suggested_skills = ["sql","communication","project management","cloud","git"]
            st.write("No direct missing skills found from the job description.")
            st.write("Recommended skills to improve for better confidence:")
            st.write(", ".join(suggested_skills))

        st.subheader("📈 Skill Match Score")
        st.write(f"{match_score:.2f}%")
    else:
        st.warning("Please upload or paste both resume text and job description.")