# app.py

"""
El-Fahman Bot - A Streamlit-based CV Analysis and Interview Assistant Application

This application provides a user interface for uploading, processing, and analyzing CVs using
AI-powered tools. It features multiple pages for different functionalities including CV upload,
processing, chatbot interaction, and CV exploration.

The application uses:
- Streamlit for the web interface
- Custom embedding models for CV processing
- AI-powered chatbot for CV analysis and interview assistance
- Qdrant vector database for storing and retrieving CV information

Main Components:
- Upload Page: For CV file uploads and preview
- Process Page: For creating embeddings and analyzing CVs
- Chat Page: For AI-powered interview assistance
- Explorer Page: For comprehensive CV exploration

Session State Variables:
    temp_pdf_paths (list): Stores paths of temporarily uploaded PDFs
    chatbot_manager (ChatbotManager): Instance of the chatbot manager
    messages (list): Stores chat history for the interview assistant
    selected_candidates (list): Stores information about selected candidates
    processing_status (int): Tracks the status of CV processing
    current_page (str): Tracks the current active page
    explorer_messages (list): Stores chat history for the CV explorer

The application is organized into four main sections:

1. Upload Page (current_page == 'upload'):
   - Handles CV file uploads
   - Provides PDF preview functionality
   - Allows multiple file selection

2. Process Page (current_page == 'process'):
   - Creates embeddings from uploaded CVs
   - Manages vector database operations
   - Handles candidate selection based on job requirements

3. Chat Page (current_page == 'chat'):
   - Provides interview assistance
   - Maintains chat history
   - Offers suggested questions

4. Explorer Page (current_page == 'explorer'):
   - Enables comprehensive CV exploration
   - Provides statistical insights
   - Allows general queries about all CVs

Each section is conditionally rendered based on the current_page session state.
"""
import streamlit as st
import time
import base64
import requests
from vectors import EmbeddingsManager
from chatbot import ChatbotManager

# Set page configuration
st.set_page_config(
    page_title="El-Fahman Bot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main > div {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #00a0dc;
        }
    </style>
""", unsafe_allow_html=True)

# Function to display PDF
def display_pdf(file):
    """
    Convert and display a PDF file in the Streamlit interface.

    Args:
        file: A file object containing the PDF to display

    Returns:
        None

    Note:
        Displays the PDF using an HTML iframe with base64 encoding
    """
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session state
if 'temp_pdf_paths' not in st.session_state:
    st.session_state['temp_pdf_paths'] = []
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'selected_candidates' not in st.session_state:
    st.session_state['selected_candidates'] = []
if 'processing_status' not in st.session_state:
    st.session_state['processing_status'] = 0
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'upload'
if 'explorer_messages' not in st.session_state:
    st.session_state['explorer_messages'] = []

# Sidebar
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.title("🎯 El-Fahman Bot")
    st.markdown("---")
    
    # Navigation with custom styling
    nav_options = {
        'upload': '📤 Upload CVs',
        'process': '⚙️ Process & Analyze',
        'chat': '💬 Interview Assistant',
        'explorer': '🔍 CV Explorer'
    }
    
    for key, label in nav_options.items():
        if st.button(
            label,
            key=f"nav_{key}",
            help=f"Navigate to {label}",
            use_container_width=True
        ):
            st.session_state['current_page'] = key
    
    st.markdown("---")
    # Status indicators
    st.subheader("📊 System Status")
    st.write(f"📑 CVs Loaded: {len(st.session_state['temp_pdf_paths'])}")
    st.write(f"🤖 AI Ready: {'✅' if st.session_state['chatbot_manager'] else '❌'}")

# Main content
if st.session_state['current_page'] == 'upload':
    st.title("📤 Upload CVs")
    
    upload_col, preview_col = st.columns([1, 2])
    
    with upload_col:
        st.markdown("""
            ### 📋 Instructions
            1. Upload multiple PDF CVs
            2. Preview them on the right
            3. Proceed to processing when ready
        """)
        
        uploaded_files = st.file_uploader(
            "Drop CVs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF files only"
        )
        
        if uploaded_files:
            st.session_state['temp_pdf_paths'] = []
            for uploaded_file in uploaded_files:
                temp_pdf_path = f"{uploaded_file.name}"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state['temp_pdf_paths'].append(temp_pdf_path)
            
            st.success(f"✅ {len(uploaded_files)} CV(s) uploaded successfully!")
            
            if st.button("➡️ Proceed to Processing", use_container_width=True):
                st.session_state['current_page'] = 'process'
    
    with preview_col:
        if uploaded_files:
            selected_cv = st.selectbox(
                "Select CV to preview",
                [file.name for file in uploaded_files]
            )
            for file in uploaded_files:
                if file.name == selected_cv:
                    display_pdf(file)

elif st.session_state['current_page'] == 'process':
    st.title("⚙️ Process & Analyze")
    
    process_col, col2 = st.columns([1, 1])
    
    with process_col:
        st.markdown("### 🔄 Processing Steps")
        
        # Create embeddings
        if st.button("1️⃣ Create CV Embeddings", disabled=not st.session_state['temp_pdf_paths']):
            embeddings_manager = EmbeddingsManager(
                model_name="BAAI/bge-small-en",
                qdrant_url="http://localhost:6333",
                collection_name="vector_db"
            )
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, pdf_path in enumerate(st.session_state['temp_pdf_paths']):
                status_text.text(f"Processing: {pdf_path}")
                embeddings_manager.create_embeddings(pdf_path)
                progress = (idx + 1) / len(st.session_state['temp_pdf_paths'])
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            status_text.text("✅ Embeddings created successfully!")
            st.session_state['processing_status'] = 1
            
            # Initialize chatbot
            st.session_state['chatbot_manager'] = ChatbotManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                llm_model="llama3.2:3b",
                llm_temperature=0.7,
                qdrant_url="http://localhost:6333",
                collection_name="vector_db"
            )

        # Delete database functionality
        st.markdown("### 🗑️ Delete Database")
        if st.button("❌ Delete Vector Database", key="delete_db_button"):
            try:
                response = requests.delete("http://localhost:6333/collections/vector_db")
                if response.status_code == 200:
                    st.success("✅ Collection 'vector_db' deleted successfully!")
                elif response.status_code == 404:
                    st.warning("⚠️ Collection 'vector_db' does not exist.")
                else:
                    st.error(f"❌ Failed to delete collection. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error while deleting collection: {e}")
    
    with col2:
        st.header("💬 Candidate Selection")
        
        # Job requirements input
        requirements = st.text_area("Enter job requirements:", height=150, help="Describe the ideal candidate profile.")
        top_n = st.number_input("Number of top candidates to retrieve:", min_value=1, step=1, value=3)
        
        if st.button("Find Top Candidates", disabled=not st.session_state['chatbot_manager']):
            st.session_state['user_requirements'] = requirements
            st.session_state['top_n'] = top_n
            query = f"Select the top {top_n} candidates based on the following requirements: {requirements}"
            
            with st.spinner("🔍 Finding candidates..."):
                try:
                    result = st.session_state['chatbot_manager'].get_response(query)
                    st.session_state['selected_candidates'] = result.split("\n")  # Save the candidates for display
                    st.success(f"✅ Successfully found {top_n} candidates!")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display selected candidates
        if st.session_state.get('selected_candidates', []):
            st.markdown("### 📋 Selected Candidates")
            st.markdown("\n".join(st.session_state['selected_candidates']))  # Display candidates as plain text

            # Button to proceed to the chatbot page
            if st.button("💬 Proceed to Chatbot"):
                st.session_state['current_page'] = 'chat'



elif st.session_state['current_page'] == 'chat':
    st.title("💬 Interview Assistant")
    
    chat_col, info_col = st.columns([2, 1])
    
    with chat_col:
        st.markdown("### 🤖 Chat with AI Assistant")
        
        # Display chat history first
        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about the selected candidates..."):
            if st.session_state['chatbot_manager']:
                st.chat_message("user").markdown(prompt)
                st.session_state['messages'].append({"role": "user", "content": prompt})
                
                with st.spinner("Thinking..."):
                    context = "\n".join(st.session_state['selected_candidates'])
                    query = f"Context: {context}\nQuestion: {prompt}"
                    try:
                        response = st.session_state['chatbot_manager'].get_response(query)
                        st.chat_message("assistant").markdown(response)
                        st.session_state['messages'].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please process CVs first!")
    
    with info_col:
        st.markdown("### 📋 Suggested Questions")
        questions = [
            "What are their key skills?",
            "Compare their experience levels.",
            "Who has the most relevant background?",
            "Rate their technical expertise.",
        ]
        for question in questions:
            if st.button(question, key=f"suggest_{question}"):
                st.chat_message("user").markdown(question)
                st.session_state['messages'].append({"role": "user", "content": question})
                
                with st.spinner("Thinking..."):
                    context = "\n".join(st.session_state['selected_candidates'])
                    query = f"Context: {context}\nQuestion: {question}"
                    try:
                        response = st.session_state['chatbot_manager'].get_response(query)
                        st.chat_message("assistant").markdown(response)
                        st.session_state['messages'].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

elif st.session_state['current_page'] == 'explorer':
    st.title("🔍 CV Explorer")
    st.markdown("### Chat with All Uploaded CVs")
    
    # Create two columns
    chat_col, info_col = st.columns([2, 1])
    
    with chat_col:
        # Display chat history
        for message in st.session_state['explorer_messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about the CVs..."):
            if st.session_state['chatbot_manager']:
                # Show user message immediately
                st.chat_message("user").markdown(prompt)
                st.session_state['explorer_messages'].append({"role": "user", "content": prompt})
                
                with st.spinner("Analyzing all CVs..."):
                    # Create a query that includes all CVs
                    query = f"Analyze all available CVs and answer: {prompt}"
                    try:
                        response = st.session_state['chatbot_manager'].get_response(query)
                        # Show assistant response immediately
                        st.chat_message("assistant").markdown(response)
                        st.session_state['explorer_messages'].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Please process CVs first!")
    
    with info_col:
        st.markdown("### 📊 CV Statistics")
        st.write(f"Total CVs: {len(st.session_state['temp_pdf_paths'])}")
        
        st.markdown("### 💡 Suggested Questions")
        example_questions = [
            "List all candidates names with Machine Learning experience",
            "Who has the most years of experience?",
            "Summarize each CV in bullet points"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"explore_{q}"):
                # Show user question immediately
                st.chat_message("user").markdown(q)
                st.session_state['explorer_messages'].append({"role": "user", "content": q})
                
                with st.spinner("Analyzing all CVs..."):
                    query = f"Analyze all available CVs and answer: {q}"
                    try:
                        response = st.session_state['chatbot_manager'].get_response(query)
                        # Show assistant response immediately
                        st.chat_message("assistant").markdown(response)
                        st.session_state['explorer_messages'].append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
        # Add a clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_explorer_chat"):
            st.session_state['explorer_messages'] = []
            st.rerun()
