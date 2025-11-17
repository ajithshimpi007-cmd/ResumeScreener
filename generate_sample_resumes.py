from docx import Document
from fpdf import FPDF
import os

os.makedirs('sample_resumes', exist_ok=True)

# Define sample resumes content
resumes = [
    {
        'filename_docx': 'sample_resumes/resume_software_engineer_1.docx',
        'filename_pdf': 'sample_resumes/resume_software_engineer_1.pdf',
        'text': '''John Doe\nSoftware Engineer\n\nSkills:\n- Python, Django, Flask, REST API\n- SQL, PostgreSQL, MySQL\n- Docker, Kubernetes, AWS\n- Git, CI/CD\n\nExperience:\nWorked on backend services using Python and Django. Built CI/CD pipelines and containerized apps.'''
    },
    {
        'filename_docx': 'sample_resumes/resume_data_scientist_2.docx',
        'filename_pdf': 'sample_resumes/resume_data_scientist_2.pdf',
        'text': '''Jane Smith\nData Scientist\n\nSkills:\n- Python, Pandas, NumPy, SciPy\n- Machine Learning, Deep Learning, TensorFlow, PyTorch\n- NLP, Natural Language Processing\n- Spark, Hadoop\n\nExperience:\nBuilt ML models for prediction and used NLP for text analytics.'''
    },
    {
        'filename_docx': 'sample_resumes/resume_frontend_3.docx',
        'filename_pdf': 'sample_resumes/resume_frontend_3.pdf',
        'text': '''Alex Johnson\nFrontend Developer\n\nSkills:\n- JavaScript, TypeScript, React, Angular, Vue\n- HTML, CSS, Responsive Design\n- REST API integration, GraphQL\n- Communication, Problem Solving\n\nExperience:\nDeveloped interactive UIs and worked with design teams to deliver responsive web apps.'''
    }
]

# Create DOCX files
for r in resumes:
    doc = Document()
    for line in r['text'].split('\n'):
        doc.add_paragraph(line)
    doc.save(r['filename_docx'])
    print('Created', r['filename_docx'])

# Create PDF files using fpdf
for r in resumes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Helvetica', size=12)
    for line in r['text'].split('\n'):
        # use a sensible cell width to avoid FPDF horizontal space errors
        pdf.multi_cell(180, 8, line)
    pdf.output(r['filename_pdf'])
    print('Created', r['filename_pdf'])

print('\nSample resumes created in sample_resumes/')
