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
from rapidfuzz import fuzz

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
    """Extract potential skills from text using spaCy noun-chunks and phrase matcher against a curated skill list."""
    text_lower = text.lower()
    doc = nlp(text_lower)

    # Base skills from noun chunks and nouns/proper nouns
    skills = set()
    for chunk in doc.noun_chunks:
        skills.add(chunk.text.strip())
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            skills.add(token.text.strip())

    # Also match against curated skill list using PhraseMatcher for accuracy (multi-word skills)
    curated = get_curated_skills()
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(s) for s in curated]
    matcher.add('SKILLS', patterns)
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end].text
        skills.add(span.strip())

    # Normalize skills: lowercase and strip
    norm = set(s.lower() for s in skills if s and len(s) > 1)

    # Fuzzy-match candidates against curated list to catch near-misses
    curated = get_curated_skills()
    candidates = set()
    for chunk in doc.noun_chunks:
        candidates.add(chunk.text.strip())
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            candidates.add(token.text.strip())

    for cand in candidates:
        cand_low = cand.lower()
        for skill in curated:
            score = fuzz.token_sort_ratio(cand_low, skill.lower())
            if score >= 85:
                norm.add(skill.lower())

    return norm

def get_curated_skills():
    """Return a curated list of common skills and multi-word phrases for better matching."""
    return [
        'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'django', 'flask', 'spring', 'sql', 'mysql', 'postgresql', 'mongodb', 'nosql',
        'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git',
        'machine learning', 'deep learning', 'nlp', 'natural language processing', 'pytorch', 'tensorflow',
        'data analysis', 'data engineering', 'pandas', 'numpy', 'scipy', 'spark', 'hadoop',
        'rest api', 'graphql', 'microservices', 'communication', 'leadership', 'problem solving'
    ]

def score_resume(resume_text, jd_text):
    """Score a resume against a job description"""
    # Preprocess texts (kept for optional overall similarity)
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)

    # Calculate overall similarity (kept as additional signal)
    try:
        similarity_score = calculate_similarity(processed_resume, processed_jd) * 100
    except Exception:
        similarity_score = 0.0

    # Extract skills using extractor
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))

    # Matched = only JD skills that are present in the resume
    matching_skills = sorted(list(jd_skills.intersection(resume_skills)))

    # Extra skills = resume skills that are NOT in JD
    extra_skills = sorted(list(resume_skills.difference(jd_skills)))

    jd_count = len(jd_skills)
    matched_count = len(matching_skills)

    # If JD has skills, skill match is fraction of JD skills covered by resume
    skills_score = (matched_count / jd_count * 100) if jd_count > 0 else 0.0

    return {
        'similarity_score': round(similarity_score, 2),
        'skills_score': round(skills_score, 2),
        'matching_skills': matching_skills,
        'extra_skills': extra_skills,
        'matched_count': matched_count,
        'jd_skill_count': jd_count,
        'extra_count': len(extra_skills)
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

# Resume Upload — one browse field under each resume section
st.subheader("Step 2: Upload up to 10 Resumes (one upload per resume section)")
st.markdown("Upload each candidate resume under its section. Leave unused sections empty.")

uploaders = []
for i in range(10):
    st.markdown(f"### Resume {i+1}")
    f = st.file_uploader(f"Browse file for Resume {i+1}", type=["pdf", "docx"], key=f"resume_{i+1}")
    uploaders.append(f)

# Analyze button
analyze_clicked = st.button("🔍 Analyze Resumes")

if jd_text and analyze_clicked:
    st.info("⏳ Analyzing resumes...")
    # Collect files from the 10 separate uploaders
    files_to_process = [f for f in uploaders if f is not None]
    if not files_to_process:
        st.warning("Please upload at least one resume (up to 10) before analyzing.")
        st.stop()
    
    try:
        results = []
        
        for file in files_to_process:
            # Extract text based on file type
            if file.name.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.name.lower().endswith('.docx'):
                text = extract_text_from_docx(file)
            else:
                continue

            # Score the resume
            scores = score_resume(text, jd_text)

            # Use the scoring output (skills are normalized inside score_resume)
            resume_skills = set(extract_skills(text))
            jd_skills = set(extract_skills(jd_text))

            matching = set(scores.get('matching_skills', []))
            extra = set(scores.get('extra_skills', []))

            results.append({
                'Resume': file.name,
                'Overall Match (%)': round(scores.get('similarity_score', 0.0), 2),
                'Skills Match (%)': round(scores.get('skills_score', 0.0), 2),
                'Matching Skills': ', '.join(sorted(matching)),
                'Resume Skills': ', '.join(sorted(resume_skills)),
                'JD Skills': ', '.join(sorted(jd_skills)),
                'Matched Count': scores.get('matched_count', len(matching)),
                'JD Skill Count': scores.get('jd_skill_count', len(jd_skills)),
                'Extra Skills Count': scores.get('extra_count', len(extra)),
                'Extra Skills': ', '.join(sorted(extra))
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

            # CSV export
            csv = df.to_csv(index=False)
            st.download_button(label="Download results as CSV", data=csv, file_name='resume_screening_results.csv', mime='text/csv')
            
            # Visualization
            st.subheader("Skills Match Visualizations")

            # Build per-resume matched/unmatched data
            jd_skill_counts = df['JD Skill Count'].tolist()
            matched_counts = df['Matched Count'].tolist()
            extra_counts = df['Extra Skills Count'].tolist()
            resumes = df['Resume'].tolist()

            # Per-resume pie charts: matched vs unmatched skills (based on JD skills)
            cols = st.columns(min(3, len(resumes)))
            for i, resume in enumerate(resumes):
                matched = matched_counts[i]
                jd_total = jd_skill_counts[i] if jd_skill_counts[i] > 0 else 0
                unmatched = jd_total - matched if jd_total > 0 else 0

                # Avoid division by zero
                if jd_total == 0:
                    with cols[i % len(cols)]:
                        st.write(f"{resume}")
                        st.info("No skills detected in JD to compute matched/unmatched skills pie.")
                    continue

                with cols[i % len(cols)]:
                    st.write(f"{resume}")
                    fig1, ax1 = plt.subplots(figsize=(4, 4))
                    sizes = [matched, unmatched]
                    labels = [f"Matched ({matched})", f"Unmatched ({unmatched})"]
                    colors = ['#4CAF50', '#FF6347']
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')
                    st.pyplot(fig1)

            # Bar chart comparing matched % and extra skills across resumes
            st.subheader("Matched / Unmatched / Extra Skills Comparison")
            matched_pct = [ (m / jd_total * 100) if jd_total>0 else 0 for m, jd_total in zip(matched_counts, jd_skill_counts) ]
            unmatched_pct = [ ( (jd - m) / jd * 100) if jd>0 else 0 for m, jd in zip(matched_counts, jd_skill_counts) ]

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            x = np.arange(len(resumes))
            width = 0.25
            ax2.bar(x - width, matched_pct, width, label='Matched %', color='#4CAF50')
            ax2.bar(x, unmatched_pct, width, label='Unmatched %', color='#FF6347')
            ax2.bar(x + width, extra_counts, width, label='Extra Skills (count)', color='#2196F3')
            ax2.set_ylabel('Percentage / Count')
            ax2.set_title('Skills Match Comparison Across Resumes')
            ax2.set_xticks(x)
            ax2.set_xticklabels(resumes, rotation=45, ha='right')
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)

            # Overall pie: across all JD skills, how many were matched at least once
            total_jd_skills = set()
            total_matched = set()
            for idx, row in df.iterrows():
                jd_set = set([s.strip().lower() for s in str(row['JD Skills']).split(',') if s.strip()])
                resume_set = set([s.strip().lower() for s in str(row['Resume Skills']).split(',') if s.strip()])
                total_jd_skills.update(jd_set)
                total_matched.update(jd_set.intersection(resume_set))

            if total_jd_skills:
                matched_overall = len(total_matched)
                unmatched_overall = len(total_jd_skills) - matched_overall
                fig3, ax3 = plt.subplots(figsize=(6, 6))
                ax3.pie([matched_overall, unmatched_overall], labels=[f'Matched ({matched_overall})', f'Unmatched ({unmatched_overall})'], colors=['#4CAF50', '#FF6347'], autopct='%1.1f%%', startangle=90)
                ax3.axis('equal')
                st.subheader('Overall JD Skill Coverage')
                st.pyplot(fig3)
            
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