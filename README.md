# El-Fahman Bot - CV Analysis & Interview Assistant

## 📋 Overview
A powerful AI-powered application for CV analysis and interview assistance, featuring automated CV processing, intelligent candidate selection, and interactive chat capabilities.

## 🚀 Features
- **CV Upload & Management**
  - Multiple PDF CV uploads
  - Real-time PDF preview
  - Batch processing capabilities

- **AI-Powered Analysis**
  - Automated CV embedding generation
  - Intelligent candidate matching
  - Advanced text analysis

- **Interactive Chat Interface**
  - AI-powered interview assistance
  - Suggested interview questions
  - Context-aware responses

- **CV Explorer**
  - Comprehensive CV search
  - Cross-CV analysis
  - Statistical insights

## 🛠️ Technical Architecture
- **Frontend**: Streamlit
- **Embedding Model**: BAAI/bge-small-en
- **Language Model**: llama3.2:3b
- **Vector Database**: Qdrant
- **Key Components**:
  - `vectors.py`: Handles CV processing and embedding generation
  - `chatbot.py`: Manages AI conversation and analysis
  - `app.py`: Streamlit interface and application logic


## 📖 Detailed Usage Guide

### 1. CV Upload Page 📤

#### Features
- **Supported Formats**: PDF files only
- **Multiple Upload**: Upload multiple CVs simultaneously
- **Preview Feature**: Select and preview individual CVs in-app

#### Best Practices
- Use clear, readable PDF files
- Ensure PDFs are not password protected
- Keep file sizes under 10MB per file
- Use consistent CV naming conventions

### 2. Processing & Analysis ⚙️

#### Creating Embeddings
1. Navigate to the Processing page
2. Click "Create CV Embeddings" button
3. Wait for processing completion
4. Monitor progress through the visual progress bar

#### Candidate Selection
1. Enter detailed job requirements in the text area
2. Specify desired number of candidates to retrieve
3. Review AI-selected candidates based on requirements
4. Click "Proceed to Interview Assistant" when ready

### 3. Interview Assistant 💬

#### Starting a Conversation
- Choose from suggested questions or type custom queries
- Focus questions on specific candidates or make comparisons
- Get AI-powered insights and analysis

#### Best Practices
- Ask specific, clear questions
- Use follow-up questions for deeper insights
- Save important conversation points
- Structure questions around key requirements

### 4. CV Explorer 🔍

#### Search Capabilities
- Perform full-text search across all uploaded CVs
- Filter candidates by:
  - Skills
  - Experience level
  - Educational background
- Compare multiple candidates simultaneously

#### Analytics Features
- View comprehensive candidate statistics
- Generate detailed comparative reports
- Export insights for external use
- Track key metrics across all candidates

## 💻 Installation

### Prerequisites
```bash
- Python 3.8+
- Qdrant running on a docker container
- Ollama running on a docker container
- Sufficient storage for CV processing
```

### Step-by-Step Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/ANBadawy/RAG-Mini-Project.git
    ```
2. Create and activate a virtual environment:
    ```bash
    # Windows
    python -m venv env
    env\Scripts\activate

    # Linux/Mac
    python3 -m venv env
    source env/bin/activate
    ```
3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Install and start Qdrant:
    ```bash
    # Using Docker
    docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant
    ```
5. Install and start Qdrant:
    ```bash
    # Using Docker
    docker pull ollama/ollama
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```
    Then Enter the container to download the model `LLama3.2:3b`
    ```bash
    docker exec -it ollama /bin/bash
    ollama pull llama3.2-3b
    ```
6. Run the application:
    ```bash
    streamlit run app.py
    ```

