# chatbot.py

import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

class ChatbotManager:
    """
    A class to manage an AI-powered chatbot specialized in CV analysis and HR-related queries.

    This class integrates various components including embedding models, language models,
    and vector stores to create an intelligent system capable of analyzing and responding
    to questions about CVs and resumes.

    Attributes:
        model_name (str): Name of the HuggingFace embedding model
        device (str): Computing device for model execution
        encode_kwargs (dict): Encoding parameters for embeddings
        llm_model (str): Name of the local language model
        llm_temperature (float): Temperature parameter for response generation
        qdrant_url (str): URL of the Qdrant vector database
        collection_name (str): Name of the vector database collection
        embeddings: HuggingFace embeddings model instance
        llm: Local language model instance
        client: Qdrant client instance
        db: Vector store instance
        qa: RetrievalQA chain instance
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:3b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
    
        """
        Initialize the ChatbotManager with all necessary components.

        Args:
            model_name (str): HuggingFace model name for embeddings.
                Defaults to "BAAI/bge-small-en".
            device (str): Computing device ('cpu' or 'cuda').
                Defaults to "cpu".
            encode_kwargs (dict): Embedding encoding parameters.
                Defaults to {"normalize_embeddings": True}.
            llm_model (str): Local LLM model name.
                Defaults to "llama3.2:3b".
            llm_temperature (float): Temperature for response generation.
                Defaults to 0.7.
            qdrant_url (str): Qdrant server URL.
                Defaults to "http://localhost:6333".
            collection_name (str): Vector database collection name.
                Defaults to "vector_db".

        Note:
            The initialization sets up a comprehensive prompt template for CV analysis
            and configures all necessary components for the QA chain.
        """

        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Local LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
            # Add other parameters if needed
        )

        # Enhanced prompt template for CV analysis
        self.prompt_template = """You are an expert HR professional and talent acquisition specialist. 
        Analyze the following CV information carefully and provide a detailed response.

        Guidelines for analysis:
        - Focus on relevant skills, experience, and qualifications
        - Consider both technical capabilities and soft skills
        - Evaluate experience levels and role relevance
        - Identify key achievements and unique qualifications
        - Maintain professional HR terminology
        - Be objective and specific in assessments

        Context from CVs: {context}

        Question: {question}

        If certain information is not available in the CVs, clearly state this rather than making assumptions.
        Provide specific examples from the CVs to support your analysis.

        Detailed professional analysis:
        """

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )


    def get_response(self, query: str) -> str:
        """
        Generate a response to a user query about CV information.

        This method processes the user's query through the QA chain, which:
        1. Retrieves relevant information from the vector store
        2. Formats the query with the specialized HR prompt
        3. Generates a detailed response using the language model

        Args:
            query (str): User's question about CV information

        Returns:
            str: Detailed response analyzing the requested CV information

        Raises:
            RuntimeError: If there's an error processing the request
        """
        try:
            response = self.qa.run(query)
            return response
        except Exception as e:
            raise RuntimeError(f"Error processing request: {e}")
