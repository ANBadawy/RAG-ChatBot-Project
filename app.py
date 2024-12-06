import streamlit as st
import os
from PDFExtractor import extract_text

st.title("PDF Text Extractor")

uploaded_files = st.file_uploader(
    "Drop PDF files here", 
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.subheader(file.name)
        text = extract_text(file)
        st.text_area("Content:", text, height=300)