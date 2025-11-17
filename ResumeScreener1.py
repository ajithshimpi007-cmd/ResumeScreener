import streamlit as st
import pandas as pd
import PyPDF2
import docx
import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import matplotlib.pyplot as plt

# Download the English language model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.info("Downloading language model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Create spaCy doc
    doc = nlp(text)
    
    # Remove stopwords and lemmatize
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    
    return ' '.join(tokens)

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using TF-IDF and cosine similarity"""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def extract_skills(text):
    """Extract potential skills from text"""
    doc = nlp(text.lower())
    # Add common technical skills patterns
    skills = set()
    
    # Extract noun phrases and proper nouns as potential skills
    for chunk in doc.noun_chunks:
        skills.add(chunk.text)
    
    # Add individual tokens that might be skills
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            skills.add(token.text)
            
    return skills

def score_resume(resume_text, jd_text):
    """Score a resume against a job description"""
    # Preprocess texts
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)
    
    # Calculate overall similarity
    similarity_score = calculate_similarity(processed_resume, processed_jd)
    
    # Extract skills
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    # Calculate skills match
    matching_skills = resume_skills.intersection(jd_skills)
    skills_score = len(matching_skills) / len(jd_skills) if jd_skills else 0
    
    return {
        'similarity_score': similarity_score * 100,
        'skills_score': skills_score * 100,
        'matching_skills': list(matching_skills)
    }
    soup = BeautifulSoup(html_file, "html.parser")
    scenarios = []
    
    # First try to find table rows with test results
    table_rows = soup.find_all('tr', class_=['passed', 'failed'])
    
    if table_rows:
        # Found table format, process table rows
        for row in table_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 5:  # Expect 5 columns: #, Test Name, Status, Duration, Error Details
                test_name = cells[1].get_text(strip=True)
                status = cells[2].get_text(strip=True)
                error_msg = cells[4].get_text(strip=True)
                
                if test_name:  # Only add if we have a test name
                    scenarios.append({
                        "Test Name": test_name,
                        "Status": status,
                        "Error Message": error_msg,
                        "Failure Type": "",
                        "Confidence Level": 0,
                        "Rationale": ""
                    })
        return pd.DataFrame(scenarios)
    
    # If no table format found, try other approaches
    scenario_elements = []
    
    # Look for test case elements
    scenario_elements.extend(soup.find_all(['tr', 'div', 'section'], 
        class_=lambda x: x and any(word in str(x).lower() for word in [
            'scenario', 'test-case', 'test', 'feature', 'case', 
            'step', 'behavior', 'example'
        ])
    ))
    
    # Look for elements with test-related text
    text_patterns = [
        'scenario:', 'test case:', 'test:', 'feature:', 
        'step:', 'given:', 'when:', 'then:', 'example:'
    ]
    scenario_elements.extend(soup.find_all(lambda tag: tag.name in ['tr', 'div', 'section'] and 
                                         any(text in tag.text.lower() for text in text_patterns)))
    
    # Add any table rows with classes
    scenario_elements.extend(soup.find_all('tr', class_=True))
    
    # Process each scenario element
    for element in scenario_elements:
        # Initialize variables with default values
        data = {
            "Test Name": "Unknown Test",
            "Status": "Unknown",
            "Error Message": "",
            "Failure Type": "",
            "Confidence Level": 0,
            "Rationale": ""
        }
        
        # Check if this is a scenario/test element
        is_scenario = False
        
        # Method 1: Check for scenario class or text
        if element.get('class') and any('scenario' in cls.lower() for cls in element.get('class')):
            is_scenario = True
        elif element.find(string=re.compile(r'Scenario:|Test case:', re.I)):
            is_scenario = True
            
        if is_scenario:
            # Extract test name using multiple strategies
            test_name = None
            
            # Strategy 1: Look for elements with specific classes
            name_element = element.find(['td', 'div', 'span', 'h1', 'h2', 'h3', 'h4'], 
                class_=lambda x: x and any(s in str(x).lower() for s in [
                    'name', 'title', 'scenario', 'test', 'case', 'description', 'feature'
                ]))
            
            # Strategy 2: Look for data attributes
            if not test_name and element.has_attr('data-title'):
                test_name = element['data-title']
            
            # Strategy 3: Try to find text with specific patterns
            name_patterns = [
                r'Scenario:[\s\n]*(.*?)(?=\n|$)',
                r'Test Case:[\s\n]*(.*?)(?=\n|$)',
                r'Test:[\s\n]*(.*?)(?=\n|$)',
                r'Feature:[\s\n]*(.*?)(?=\n|$)',
                r'Given[\s\n]*(.*?)(?=\n|$)',
                r'When[\s\n]*(.*?)(?=\n|$)',
                r'Then[\s\n]*(.*?)(?=\n|$)'
            ]
            
            if name_element:
                test_name = name_element.text.strip()
            else:
                element_text = element.text.strip()
                for pattern in name_patterns:
                    match = re.search(pattern, element_text, re.IGNORECASE)
                    if match:
                        test_name = match.group(1).strip()
                        break
                
                # If still no match, try first significant text
                if not test_name:
                    for text in element.stripped_strings:
                        if len(text.strip()) > 5:  # Avoid very short strings
                            test_name = text.strip()
                            break
            
            # Clean up test name if found
            if test_name:
                # Remove common prefixes
                test_name = re.sub(r'^(Scenario|Test Case|Test|Feature|Given|When|Then):\s*', '', test_name, flags=re.IGNORECASE)
                test_name = test_name.strip()
                if test_name:
                    data["Test Name"] = test_name
            
            # Extract status using multiple approaches
            status_found = False
            
            # Approach 1: Look for status in class names
            status_classes = {
                'pass': ['passed', 'success', 'ok', 'green'],
                'fail': ['failed', 'failure', 'error', 'red']
            }
            
            element_classes = ' '.join(element.get('class', [])).lower()
            for status, class_indicators in status_classes.items():
                if any(indicator in element_classes for indicator in class_indicators):
                    data["Status"] = 'Passed' if status == 'pass' else 'Failed'
                    status_found = True
                    break
            
            # Approach 2: Look for status elements
            if not status_found:
                status_element = element.find(['td', 'span', 'div', 'i'], 
                    class_=lambda x: x and any(s in str(x).lower() for s in [
                        'status', 'result', 'outcome', 'state', 'icon'
                    ]))
                if status_element:
                    status_text = status_element.text.strip().lower()
                    if any(word in status_text for word in ['pass', 'passed', 'success', 'ok']):
                        data["Status"] = 'Passed'
                        status_found = True
                    elif any(word in status_text for word in ['fail', 'failed', 'error', 'broken']):
                        data["Status"] = 'Failed'
                        status_found = True
            
            # Approach 3: Look for status indicators in full text
            if not status_found:
                element_text = element.text.lower()
                if any(word in element_text for word in ['✓', '✔', 'pass', 'passed', 'success', 'ok', 'succeeded']):
                    data["Status"] = 'Passed'
                elif any(word in element_text for word in ['✗', '✘', 'fail', 'failed', 'error', 'broken']):
                    data["Status"] = 'Failed'
            
            # Extract error message with improved detection
            error_containers = [element]
            error_containers.extend(element.find_next_siblings(['tr', 'div', 'pre', 'section'], limit=3))
            
            for container in error_containers:
                # Look for error messages in various formats
                error_selectors = [
                    (['td', 'div', 'pre', 'span'], {'class': lambda x: x and any(s in str(x).lower() for s in [
                        'error', 'failure', 'stacktrace', 'message', 'exception', 'detail'
                    ])}),
                    (['pre', 'code'], {}),  # Any pre or code block
                    (['div', 'span'], {'style': lambda x: x and 'color: red' in str(x).lower()})  # Red text
                ]
                
                for tags, attrs in error_selectors:
                    error_element = container.find(tags, **attrs)
                    if error_element:
                        error_text = error_element.text.strip()
                        if error_text and len(error_text) > 5:  # Avoid very short messages
                            data["Error Message"] = error_text
                            break
                
                if data["Error Message"]:
                    break
            
            # Only add if we have a valid test name
            if data["Test Name"] != "Unknown Test":
                scenarios.append(data)
    return pd.DataFrame(scenarios)

def classify_failure_local(test_name: str, status: str, error_msg: str):
    """Classify a test result based on status and error messages.

    Categories:
    - Passed (Dark GREEN): When status contains pass/success
    - Valid Application Defect (RED): Failed tests with assertion errors
    - Automation Script Issue (ORANGE): Failed tests with other errors/exceptions
    """
    s = str(status or "").lower().strip()
    e = str(error_msg or "").lower().strip()

    # Pass check (Dark GREEN)
    if any(k in s for k in ["pass", "passed", "ok", "success", "succeeded", "done"]):
        return "Passed", 100, "Test passed successfully"

    # Application Defect check (RED)
    # Check for assertion errors and verification failures
    assertion_patterns = [
        'assertionerror',
        'assert',
        'expected',
        'but got',
        'verification failed',
        'expected value',
        'actual value',
        'should be',
        'should have been'
    ]
    if any(pattern in e for pattern in assertion_patterns):
        details = str(error_msg) if error_msg else "No detailed error message available"
        return "Valid Application Defect", 100, f"Assertion/Verification failure: {details}"
    if any(k in s for k in ["fail", "failed", "error", "broken"]) or e:
        # Check for common automation script issues
        automation_keywords = [
            "nosuchelement", "no such element", "element not found", 
            "unable to locate", "timeout", "timed out",
            "staleelementreference", "stale element", 
            "element not interactable", "selenium.common.exceptions",
            "selenium", "attributeerror", "valuerror",
            "exception", "error", "traceback",
            "undefined", "null pointer", "type error",
            "syntax error", "runtime error"
        ]
        
        if any(k in e.lower() for k in automation_keywords):
            details = error_msg if error_msg else "No detailed error message available"
            return ("Automation Script Issue", 90, f"Automation related error detected: {details}")
        
        # If it's a failure but doesn't match known patterns, still categorize as automation issue
        return ("Automation Script Issue", 75, f"Unspecified failure: {error_msg}" if error_msg else "Unknown error")
    
    # Default case - if we can't categorize it but there's some indication of issue
    if error_msg:
        return ("Automation Script Issue", 60, f"Uncategorized issue: {error_msg}")
    
    return ("Automation Script Issue", 50, "Status unclear or unknown issue")

st.title("Resume Screener")
st.markdown("""
This application helps you screen resumes against a job description to find the best matches.
""")

def color_score(score):
    """Color formatting for scores"""
    if score >= 80:
        return 'background-color: #90EE90'  # Light green
    elif score >= 60:
        return 'background-color: #FFFFE0'  # Light yellow
    else:
        return 'background-color: #FFB6C1'  # Light red

# Job Description Input
st.subheader("Step 1: Enter Job Description")
jd_text = st.text_area("Enter the job description", height=200)

# Resume Upload
st.subheader("Step 2: Upload Resumes")
uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)

# Analyze button
analyze_clicked = st.button("🔍 Analyze Resumes")

if jd_text and uploaded_files and analyze_clicked:
    st.info("⏳ Analyzing resumes...")
    
    try:
        results = []
        
        for file in uploaded_files:
            # Extract text based on file type
            if file.name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.name.lower().endswith('.docx'):
                text = extract_text_from_docx(file)
            else:
                continue
                
            # Score the resume
            scores = score_resume(text, jd_text)
            
            results.append({
                'Resume': file.name,
                'Overall Match (%)': round(scores['similarity_score'], 2),
                'Skills Match (%)': round(scores['skills_score'], 2),
                'Matching Skills': ', '.join(scores['matching_skills']),
            })
            
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Calculate average score
            df['Average Score'] = df[['Overall Match (%)', 'Skills Match (%)']].mean(axis=1)
            
            # Sort by average score
            df = df.sort_values('Average Score', ascending=False)
            
            # Create styled dataframe
            styled_df = df.style.applymap(
                lambda x: color_score(x) if isinstance(x, (int, float)) else '',
                subset=['Overall Match (%)', 'Skills Match (%)', 'Average Score']
            )
            
            # Display results
            st.success("✅ Analysis complete!")
            
            # Show results overview
            st.subheader("Resume Analysis Results")
            st.dataframe(styled_df)
            
            # Visualization
            st.subheader("Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            x = range(len(df))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], df['Overall Match (%)'], 
                  width, label='Overall Match', color='skyblue')
            ax.bar([i + width/2 for i in x], df['Skills Match (%)'], 
                  width, label='Skills Match', color='lightgreen')
            
            ax.set_ylabel('Score (%)')
            ax.set_title('Resume Scores Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Resume'], rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Detailed Analysis for each resume
            st.subheader("Detailed Analysis")
            for idx, row in df.iterrows():
                with st.expander(f"📄 {row['Resume']} - Average Score: {row['Average Score']:.2f}%"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Overall Match", f"{row['Overall Match (%)']}%")
                    with col2:
                        st.metric("Skills Match", f"{row['Skills Match (%)']}%")
                    st.markdown("**Matching Skills:**")
                    st.write(row['Matching Skills'])
            
        else:
            st.warning("No valid resumes found to analyze. Please upload PDF or DOCX files.")
            
    except Exception as e:
        st.error(f"Error analyzing resumes: {str(e)}")
        st.stop()