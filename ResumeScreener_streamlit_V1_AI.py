import streamlit as st
st.set_page_config(page_title="Resume Skill match Screener", layout="wide")  # <-- FIRST Streamlit command

import pandas as pd
import json
from ollama import Client
import matplotlib.pyplot as plt
import os
import io
from io import BytesIO
import re

client = Client()

def extract_skills(text):
    """Tokenize input text into word tokens (lowercased).

    NOTE: This helper extracts individual word tokens from free text.
    For job description skills we will parse phrases (see parse_job_skills).
    """
    if not text:
        return set()
    tokens = re.findall(r"\w+", text.lower())
    return set(tokens)

def extract_resume_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    text = ""
    if ext == ".txt":
        text = file.read().decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            text = ""
    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception:
            text = ""
    return text

def analyze_resume_with_llama(job_desc, resume_text):
    """Use Llama3 to analyze resume against job description."""
    prompt = f"""
You are an expert HR recruiter. Analyze the following job description and resume.

First, extract the key skills/keywords from the job description.

Then, identify which of those skills/keywords are present in the resume (even if worded differently but mean the same).

Job Description:
{job_desc}

Resume Text:
{resume_text}

Respond in JSON format:
{{
    "extracted_skills": ["skill1", "skill2", ...],
    "matched": ["skill1", "skill2", ...],
    "unmatched": ["skill3", "skill4", ...]
}}

Only output valid JSON, no additional text.
"""
    try:
        response = client.chat(model='llama3', messages=[{"role": "user", "content": prompt}])
        content = response['message']['content'].strip()
        # Remove any markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        if not content:
            raise ValueError("Empty response")
        result = json.loads(content)
        matched = result.get("matched", [])
        unmatched = result.get("unmatched", [])
        return matched, unmatched
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        st.error(f"Error parsing AI response: {str(e)}. Using fallback matching.")
        # Fallback: simple keyword extraction and matching
        job_words = set(re.findall(r'\b\w+\b', job_desc.lower()))
        resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
        matched = list(job_words & resume_words)
        unmatched = list(job_words - resume_words)
        return matched, unmatched

# styling
st.markdown("""
<style>
/* Page background and font */
body { background-color: #f7f9fb; }
.reportview-container .main header {background-color: #0b5cff}
.big-title {font-size:28px; font-weight:700; color:#0b3d91}
.subtitle {font-size:14px; color:#444}
.card {background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
.small {font-size:12px; color:#666}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='big-title'>Resume Screener - Skill Match Visualizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload up to 10 resumes and enter the job description. AI analyzes matches and provides per-resume visuals and a downloadable Excel report.</div>", unsafe_allow_html=True)

# job description input
job_desc = st.text_area("Enter Job Description", "", height=120)

uploaded_files = []
for i in range(1, 11):
    uploaded_file = st.file_uploader(f"Browse Resume {i}", type=["pdf", "docx", "txt"], key=f"resume_{i}")
    if uploaded_file:
        uploaded_files.append(uploaded_file)

if st.button("Screen Resumes"):
    if not job_desc.strip():
        st.error("Please enter a job description.")
    elif len(uploaded_files) == 0:
        st.error("Please upload at least one resume (up to 10).")
    else:
        match_percents = []
        resume_names = []
        matched_skills_list = []
        unmatched_skills_list = []
        matched_counts = []
        unmatched_counts = []
        for file in uploaded_files:
            resume_text = extract_resume_text(file)
            if not resume_text:
                st.error(f"Could not extract text from {file.name}")
                continue
            matched, unmatched = analyze_resume_with_llama(job_desc, resume_text)
            matched_count = len(matched)
            unmatched_count = len(unmatched)
            total_skills = matched_count + unmatched_count
            percent = int((matched_count / total_skills) * 100) if total_skills > 0 else 0
            match_percents.append(percent)
            matched_counts.append(matched_count)
            unmatched_counts.append(unmatched_count)
            resume_names.append(file.name)
            matched_skills_list.append(", ".join(matched) if matched else "-")
            unmatched_skills_list.append(", ".join(unmatched) if unmatched else "-")

        # Summary header card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Skill Match Report")
        # Prepare grid data dynamically based on uploaded files
        matched_grid = []
        unmatched_grid = []
        for idx in range(len(resume_names)):
            matched_grid.append({
                "Resume": resume_names[idx],
                "Matched Words": matched_skills_list[idx],
                "Matched Count": matched_counts[idx]
            })
            unmatched_grid.append({
                "Resume": resume_names[idx],
                "Unmatched Words": unmatched_skills_list[idx],
                "Unmatched Count": unmatched_counts[idx]
            })

        st.markdown("### Matched Words Grid")
        st.table(matched_grid)
        st.markdown("### Unmatched Words Grid")
        st.table(unmatched_grid)

        # Per-resume detailed view with pie chart
        st.markdown("## Per-resume details")
        for idx in range(len(resume_names)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {resume_names[idx]}")
                st.write(f"Match %: {match_percents[idx]}% ({matched_counts[idx]} matched, {unmatched_counts[idx]} unmatched)")
                st.write("**Matched Words:**")
                st.write(matched_skills_list[idx])
                st.write("**Unmatched Words:**")
                st.write(unmatched_skills_list[idx])
            with col2:
                sizes = [matched_counts[idx], unmatched_counts[idx]]
                labels = ["Matched", "Unmatched"]
                fig, ax = plt.subplots(figsize=(3, 3))
                if sum(sizes) > 0:
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
                    ax.set_title('Match vs Unmatched')
                else:
                    ax.text(0.5, 0.5, 'No tokens to compare', ha='center', va='center')
                    ax.axis('off')
                st.pyplot(fig)

        # Overall bar chart of match percentages
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.bar(resume_names, match_percents, color='skyblue')
        ax2.set_ylabel('Match %')
        ax2.set_title('Resume Match % (per resume)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        # Close summary card
        st.markdown("</div>", unsafe_allow_html=True)

        # Build overall summary dataframe and provide download as Excel
        summary_rows = []
        for i, name in enumerate(resume_names):
            status = "SELECTED" if match_percents[i] >= 75 else "NOT SELECTED"
            summary_rows.append({
                'Resume': name,
                'Matched Count': matched_counts[i],
                'Matched Words': matched_skills_list[i],
                'Unmatched Count': unmatched_counts[i],
                'Unmatched Words': unmatched_skills_list[i],
                'Match %': match_percents[i],
                'STATUS': status
            })
        summary_df = pd.DataFrame(summary_rows)
        # Add aggregate totals
        total_matches = summary_df['Matched Count'].sum()
        total_unmatches = summary_df['Unmatched Count'].sum()
        avg_match = summary_df['Match %'].mean() if not summary_df.empty else 0
        agg_df = pd.DataFrame([{
            'Total Matched': total_matches,
            'Total Unmatched': total_unmatches,
            'Average Match %': round(avg_match, 1)
        }])

        # Create Excel with multiple sheets
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(matched_grid).to_excel(writer, sheet_name='Matched', index=False)
            pd.DataFrame(unmatched_grid).to_excel(writer, sheet_name='Unmatched', index=False)
            agg_df.to_excel(writer, sheet_name='Totals', index=False)
        towrite.seek(0)
        b64 = towrite.getvalue()
        st.download_button(label='Download full report as Excel',
                           data=b64,
                           file_name='resume_match_report.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
