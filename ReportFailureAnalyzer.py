import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from ollama import Client
import json

client = Client()

def parse_selenium_html_report(html_file):
    soup = BeautifulSoup(html_file, "html.parser")
    rows = soup.find_all("tr")
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 3:
            test_name = cols[0].text.strip()
            status = cols[1].text.strip()
            error_msg = cols[2].text.strip()
            data.append({"Test Name": test_name, "Status": status, "Error Message": error_msg})
    return pd.DataFrame(data)

def llama3_classify_failure(test_name, status, error_msg):
    prompt = [
        {'role': 'system', 'content': 'You are a QA automation expert. Respond ONLY in JSON format.'},
        {'role': 'user', 'content': f"""
Analyze the following Selenium test result and respond strictly in JSON format:

Test Name: "{test_name}"
Status: "{status}"
Error Message: "{error_msg}"

Respond with:
{{
  "Failure Type": "Valid Application Defect / Script Failure / Passed / Needs Review",
  "Confidence Level": 0-100,
  "Rationale": "Brief explanation"
}}

Guidelines:
- Return only valid JSON. No markdown, bullet points, or extra commentary.
- Use only the failure types listed above.
"""}
    ]
    try:
        response = client.chat(model='llama3', messages=prompt)
        content = response['message']['content'].strip()
        result = json.loads(content)
        return (
            result.get("Failure Type", "Needs Review"),
            int(result.get("Confidence Level", 60)),
            result.get("Rationale", "")
        )
    except Exception as e:
        return ("Needs Review", 60, f"Model error: {str(e)}")

st.title("Automation Report Failures Analyzer (Llama3 Powered)")

uploaded_file = st.file_uploader("Upload HTML Report", type=["html"])
analyze_clicked = st.button("🔍 Analyze")
if analyze_clicked:
    st.info("⏳ Running analysis...")

if uploaded_file and analyze_clicked:
    df = parse_selenium_html_report(uploaded_file)
    df[["Failure Type", "Confidence Level", "Rationale"]] = df.apply(
        lambda row: llama3_classify_failure(row["Test Name"], row["Status"], row["Error Message"]),
        axis=1, result_type="expand"
    )

    st.success("✅ Report analyzed!")

    # 4 Quadrants Visuals
    st.subheader("Test Failure Analysis (4 Quadrants View)")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown("**Failure Type Distribution**")
        st.bar_chart(df["Failure Type"].value_counts())

    with col2:
        st.markdown("**Confidence Level by Test Case**")
        st.line_chart(df[["Test Name", "Confidence Level"]].set_index("Test Name"))

    with col3:
        st.markdown("**Application Defects**")
        st.dataframe(df[df["Failure Type"] == "Valid Application Defect"][["Test Name", "Error Message", "Confidence Level", "Rationale"]])

    with col4:
        st.markdown("**Script Failures**")
        st.dataframe(df[df["Failure Type"] == "Script Failure"][["Test Name", "Error Message", "Confidence Level", "Rationale"]])

    st.subheader("Detailed Test Case Analysis")
    st.dataframe(df)
