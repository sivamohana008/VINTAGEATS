import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai  # Correct import
import PyPDF2

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini (use your API key)
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-2.5-flash')  # Correct model access

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# PDF PARSING
# ==============================
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# ==============================
# RESUME PARSER (LLM)
# ==============================
def parse_resume(resume_text):
    prompt = f"""
You are a resume parser.

Extract:
- Skills
- Experience summary
- Education
- Tools & technologies

Resume:
{resume_text}

Return in bullet points.
"""
    response = model.generate_content(prompt)
    return response.text

# ==============================
# JOB DESCRIPTION PARSER
# ==============================
def parse_job_description(jd_text):
    prompt = f"""
Extract:
- Required skills
- Responsibilities
- Preferred qualifications

Job Description:
{jd_text}

Return in bullet points.
"""
    response = model.generate_content(prompt)
    return response.text

# ==============================
# ATS MATCHING
# ==============================
def ats_match(parsed_resume, parsed_jd):
    prompt = f"""
You are an Applicant Tracking System.

Compare the resume and job description.

Resume:
{parsed_resume}

Job Description:
{parsed_jd}

Provide:
1. Match percentage (0-100)
2. Matching skills
3. Missing skills
4. Strengths
5. Improvement suggestions
"""
    response = model.generate_content(prompt)
    return response.text

@app.route('/')
def home():
    return render_template("index.html")

# ==============================
# API ROUTE (PDF UPLOAD)
# ==============================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")

    if not jd_text:
        return jsonify({"error": "Job description is required"}), 400

    # Save PDF
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_file.filename)
    resume_file.save(pdf_path)

    # Extract resume text
    resume_text = extract_text_from_pdf(pdf_path)

    # Parse using Gemini
    parsed_resume = parse_resume(resume_text)
    parsed_jd = parse_job_description(jd_text)

    # ATS Matching
    ats_result = ats_match(parsed_resume, parsed_jd)

    return jsonify({
        "parsed_resume": parsed_resume,
        "parsed_job_description": parsed_jd,
        "ats_result": ats_result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
