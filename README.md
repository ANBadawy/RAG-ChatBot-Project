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
