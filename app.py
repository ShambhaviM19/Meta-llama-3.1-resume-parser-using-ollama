import streamlit as st
from PyPDF2 import PdfReader
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

st.title("Resume Parser using Langchain and Llama3.1")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Prompt template
template = """
Resume Text: {resume_text}

Extract the following details and return them ONLY in JSON format without any additional text:
- Name
- Email
- Phone Number
- Summary
- Current Location
- Current Company
- Skills
- Linkedin ID
- Github ID
- Total Experience
- Education (Degree, Specialization, Institute, Start Year, End Year)
- Education Year
- Experiences (Company Name, Designation, Start Date, End Date, Description)
- Projects (Project, Project Description)
- Roles and Responsibilities
- Certifications

Provide the response in the following JSON format:
{{
  "Name": "",
  "Email": "",
  "Phone-Number": "",
  "Summary": "",
  "Current-Location": "",
  "Current-Company": "",
  "Skills": [],
  "Linkedin-Id": "",
  "Github-Id": "",
  "Total-Experience": 0,
  "Education": [],
  "Education-Year": [],
  "Experiences": [],
  "Projects": [],
  "Roles-Responsibility": [],
  "Certifications": []
}}

Return ONLY the JSON object without any additional text or explanation.
"""

prompt = ChatPromptTemplate.from_template(template)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model = OllamaLLM(
    model="llama3.1",
    callback_manager=callback_manager,
    request_timeout=300
)

chain = prompt | model

uploaded_file = st.file_uploader("Upload your resume in PDF format", type="pdf")
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Parsing resume..."):
        try:
            result = chain.invoke({"resume_text": pdf_text})
        except Exception as e:
            st.error(f"An error occurred while processing the resume: {str(e)}")
            st.stop()
    try:
        parsed_information = json.loads(result)
        st.write("Parsed Information:", parsed_information)
    except json.JSONDecodeError:
        st.error("Failed to parse the resume text into the required JSON format.")
        st.text(result)