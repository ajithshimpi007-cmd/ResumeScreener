import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt

def parse_selenium_html_report(html_file):
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

st.title("Automation Report Failures Analyzer")

def color_status(row):
    """Color code the rows based on failure type:
    - Light Red for Valid Application Defect
    - Light Green for Passed
    - Light Orange for Automation Script Issues
    """
    if row["Failure Type"] == "Valid Application Defect":
        return ['background-color: #FFE6E6'] * len(row)  # Light red
    elif row["Failure Type"] == "Passed":
        return ['background-color: #E6FFE6'] * len(row)  # Light green
    else:
        return ['background-color: #FFE9CC'] * len(row)  # Light orange

uploaded_file = st.file_uploader("Upload HTML Report", type=["html"])
analyze_clicked = st.button("🔍 Analyze")
if analyze_clicked:
    st.info("⏳ Running analysis...")

if uploaded_file and analyze_clicked:
    st.info("⏳ Running analysis...")
    
    try:
        # Parse the HTML report
        df = parse_selenium_html_report(uploaded_file)
        
        if df.empty:
            st.error("No test cases found in the report. Please check the HTML report format.")
            
            # Debug information to help identify the issue
            soup = BeautifulSoup(uploaded_file, "html.parser")
            st.warning("Debug Information:")
            st.write("Document structure found:")
            st.write("- Total elements:", len(soup.find_all()))
            st.write("- Tables found:", len(soup.find_all('table')))
            st.write("- Table rows found:", len(soup.find_all('tr')))
            st.write("- Divs with 'test' or 'scenario' in class:", 
                    len(soup.find_all('div', class_=lambda x: x and ('test' in str(x).lower() or 'scenario' in str(x).lower()))))
            
            st.info("Try uploading a different HTML report or contact support for assistance.")
            st.stop()
        
        # Verify all required columns exist
        required_columns = ["Test Name", "Status", "Error Message", "Failure Type", "Confidence Level"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing columns in the report: {', '.join(missing_columns)}")
            st.stop()
        
        # Apply classification row by row
        for idx, row in df.iterrows():
            failure_type, confidence, rationale = classify_failure_local(
                row["Test Name"], 
                row["Status"], 
                row["Error Message"]
            )
            df.at[idx, "Failure Type"] = failure_type
            df.at[idx, "Confidence Level"] = confidence
            df.at[idx, "Rationale"] = rationale
            
        st.success("✅ Report analyzed successfully!")
    except Exception as e:
        st.error(f"Error analyzing report: {str(e)}")
        st.stop()

    st.success("✅ Report analyzed!")

    # Overall Results with Color Coding
    st.subheader("Test Results Overview")
    styled_df = df[["Test Name", "Status", "Error Message", "Failure Type", "Confidence Level"]].style.apply(color_status, axis=1)
    st.dataframe(styled_df)

    # Test Results Distribution
    st.subheader("Test Results Distribution")
    
    # Calculate counts and percentages
    n_defects = len(df[df["Failure Type"] == "Valid Application Defect"])
    n_passed = len(df[df["Failure Type"] == "Passed"])
    n_other = len(df[~df["Failure Type"].isin(["Valid Application Defect", "Passed"])])
    total = len(df)
    
    # Create pie chart with percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Pie Chart
    sizes = [n_defects, n_passed, n_other]
    labels = ['Application Defects', 'Passed Tests', 'Automation Script Issues']
    colors = ['#ff0000', '#006400', '#FFA500']  # Dark Red, Dark Green, Orange
    
    # Avoid division by zero
    total = max(total, 1)  # Ensure total is at least 1
    percentages = [count/total*100 for count in sizes]
    
    wedges, texts, autotexts = ax1.pie(sizes, 
                                      labels=labels, 
                                      colors=colors,
                                      autopct='%1.1f%%',
                                      startangle=90)
    ax1.axis('equal')
    plt.setp(autotexts, size=14, weight="bold")  # Increased from 8 to 14
    plt.setp(texts, size=12)  # Increased from 10 to 12
    ax1.set_title("Test Results Distribution", pad=20)

    # Bar chart for counts
    bars = ax2.bar(labels, sizes, color=colors)
    ax2.set_title("Test Results Count")
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Detailed breakdowns in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error(f"🐛 Application Defects: {n_defects}")
        st.markdown(f"<h2 style='text-align: center; color: #CD5C5C;'>{n_defects/total*100:.1f}%</h2>", unsafe_allow_html=True)
        if n_defects > 0:
            st.markdown("**Application Defects (Red)**")
            defects_df = df[df["Failure Type"] == "Valid Application Defect"][["Test Name", "Error Message"]]
            st.dataframe(defects_df)

    with col2:
        st.success(f"✅ Passed Tests: {n_passed}")
        st.markdown(f"<h2 style='text-align: center; color: #2E8B57;'>{n_passed/total*100:.1f}%</h2>", unsafe_allow_html=True)
        if n_passed > 0:
            st.markdown("**Passed Tests (Green)**")
            passed_df = df[df["Failure Type"] == "Passed"][["Test Name", "Status"]]
            st.dataframe(passed_df)

    with col3:
        st.warning(f"⚠️ Automation Script Issues: {n_other}")
        st.markdown(f"<h2 style='text-align: center; color: #FFA500;'>{n_other/total*100:.1f}%</h2>", unsafe_allow_html=True)
        if n_other > 0:
            st.markdown("**Automation Script Issues (Orange)**")
            other_df = df[~df["Failure Type"].isin(["Valid Application Defect", "Passed"])][["Test Name", "Error Message"]]
            st.dataframe(other_df)

    st.subheader("Detailed Test Case Analysis")
    st.dataframe(df)
