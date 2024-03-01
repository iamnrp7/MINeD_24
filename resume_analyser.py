import streamlit as st
import pdfplumber
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

custom_theme = {
    "primaryColor": "#ff6f61",  # Coral color
    "backgroundColor": "#f8edeb",  # Light peach background
    "secondaryBackgroundColor": "#ffe4e1",  # Light pink secondary background
    "textColor": "#2c3e50",  # Dark text color
    "font": "sans-serif",
}

# Set the custom theme
st.set_page_config(
    page_title="RESUME ANALYSER",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply the custom theme
st.markdown(
    f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 1200px;
                padding-top: 2rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
            }}
        </style>
    """,
    unsafe_allow_html=True,
)



# Initialize Gemini LLM with Google API key
google_api_key = "AIzaSyDX7iqE8XTN8npHp9jKZST8HMfZS4ncNpg"  # Replace with your Google API key
llm = GoogleGenerativeAI(temperature=0.1, google_api_key=google_api_key, model="gemini-pro")

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['text'],
    template="Please provide a rewritten version of {text}."
)

second_input_prompt = PromptTemplate(
    input_variables=['descript'],
    template="Please extract and provide education details from {descript}."
)

# Chain of LLMs
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='descript')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dis_two')

# Parent Chain
parent_chain = SequentialChain(chains=[chain1, chain2], input_variables=['text'], output_variables=['descript', 'dis_two'], verbose=True)


# Streamlit UI
st.title('RESUME ANALYSER')

# File uploader for resume PDF
uploaded_file = st.file_uploader("Upload Resume PDF", type=['pdf'])

if uploaded_file is not None:
    # Extract text from uploaded PDF
    with pdfplumber.open(uploaded_file) as pdf:
        extracted_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            extracted_text += page_text + "\n"

    # Display extracted text
    st.subheader("Extracted Text from PDF:")
    
   
    # Execute the LLM chain using the extracted text
    if extracted_text:
        try:
            result = parent_chain({'text': extracted_text})
          
            with st.expander("Information"):
                st.write(result['dis_two'])
            st.write('Data Fetched Successfully')
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to analyze.")



