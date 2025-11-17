from docx import Document
from fpdf import FPDF
import os

os.makedirs('sample_resumes', exist_ok=True)

additional = [
    {
        'docx': 'sample_resumes/resume_backend_4.docx',
        'pdf': 'sample_resumes/resume_backend_4.pdf',
        'text': '''Maria Gomez\nBackend Engineer\n\nSkills:\n- Java, Spring Boot, REST API\n- MySQL, MongoDB\n- Docker, Kubernetes\n- Git, Jenkins, CI/CD\n\nExperience:\nBuilt microservices using Java and Spring Boot for high-traffic APIs.'''
    },
    {
        'docx': 'sample_resumes/resume_devops_5.docx',
        'pdf': 'sample_resumes/resume_devops_5.pdf',
        'text': '''Liam Brown\nDevOps Engineer\n\nSkills:\n- AWS, Azure, GCP\n- Docker, Kubernetes, Terraform\n- CI/CD, Jenkins, GitLab CI\n- Monitoring: Prometheus, Grafana\n\nExperience:\nImplemented infrastructure as code and deployment pipelines.'''
    },
    {
        'docx': 'sample_resumes/resume_mobile_6.docx',
        'pdf': 'sample_resumes/resume_mobile_6.pdf',
        'text': '''Olivia Davis\nMobile Developer\n\nSkills:\n- Swift, Kotlin, React Native\n- REST APIs, GraphQL\n- Unit Testing, CI/CD\n- Communication, Teamwork\n\nExperience:\nDeveloped cross-platform mobile apps and integrated APIs.'''
    },
    {
        'docx': 'sample_resumes/resume_qa_7.docx',
        'pdf': 'sample_resumes/resume_qa_7.pdf',
        'text': '''Ethan Wilson\nQA Engineer\n\nSkills:\n- Selenium, Playwright, Automation Testing\n- Python, Java\n- Test Plans, Test Cases, Reporting\n- Jira, TestRail\n\nExperience:\nAutomated regression suites and coordinated test plans.'''
    }
]

for r in additional:
    doc = Document()
    for line in r['text'].split('\n'):
        doc.add_paragraph(line)
    doc.save(r['docx'])
    print('Created', r['docx'])

for r in additional:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Helvetica', size=12)
    for line in r['text'].split('\n'):
        pdf.multi_cell(180, 8, line)
    pdf.output(r['pdf'])
    print('Created', r['pdf'])

print('\nAdditional sample resumes created in sample_resumes/')
