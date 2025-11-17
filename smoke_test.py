import os
import glob
import pandas as pd
from ResumeScreener import extract_text_from_pdf, extract_text_from_docx, score_resume

SAMPLE_DIR = 'sample_resumes'
JD_TEXT = '''Software Engineer with Python experience\nRequired skills:\n- Python programming\n- Data structures and algorithms\n- Web frameworks (Django/Flask)\n- Database management\n- Git version control\n'''

files = sorted(glob.glob(os.path.join(SAMPLE_DIR, '*')))
# Take up to 10 files
files = files[:10]

results = []
for f in files:
    name = os.path.basename(f)
    if f.lower().endswith('.pdf'):
        text = extract_text_from_pdf(open(f, 'rb'))
    elif f.lower().endswith('.docx'):
        text = extract_text_from_docx(f)
    else:
        continue
    scores = score_resume(text, JD_TEXT)
    results.append({'Resume': name, 'Overall Match (%)': scores['similarity_score'], 'Skills Match (%)': scores['skills_score'], 'Matching Skills': ', '.join(scores['matching_skills'])})

if results:
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('sample_smoke_results.csv', index=False)
    print('\nWrote sample_smoke_results.csv')
else:
    print('No results')
