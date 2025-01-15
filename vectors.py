# vectors.py

import os
import base64
import uuid
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
import requests
from qdrant_client.http import models



class EmbeddingsManager:
    """
    A class to manage the creation and storage of document embeddings using HuggingFace models and Qdrant vector database.

    This class handles the process of loading PDF documents, creating text embeddings using HuggingFace models,
    and storing them in a Qdrant vector database for later retrieval and similarity search.

    Attributes:
        model_name (str): Name of the HuggingFace model to use for embeddings
        device (str): Computing device to use ('cpu' or 'cuda')
        encode_kwargs (dict): Additional encoding parameters
        qdrant_url (str): URL of the Qdrant server
        collection_name (str): Name of the collection in Qdrant
        embeddings: HuggingFaceBgeEmbeddings instance for creating embeddings
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = f"vector_db{uuid.uuid4().hex}",
    ):
        """
        Initialize the EmbeddingsManager with specified parameters.

        Args:
            model_name (str): HuggingFace model name for embeddings generation.
                Defaults to "BAAI/bge-small-en".
            device (str): Computing device to use ('cpu' or 'cuda').
                Defaults to "cpu".
            encode_kwargs (dict): Additional encoding parameters.
                Defaults to {"normalize_embeddings": True}.
            qdrant_url (str): URL of the Qdrant server.
                Defaults to "http://localhost:6333".
            collection_name (str): Name of the collection in Qdrant.
                Defaults to a randomly generated name.

        Returns:
            None
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, pdf_path: str):
        """
        Process a PDF document, create embeddings, and store them in Qdrant.

        This method performs the following steps:
        1. Loads and validates the PDF document
        2. Extracts the candidate's name from the filename
        3. Splits the document into manageable chunks
        4. Adds candidate information to each chunk
        5. Creates embeddings and stores them in Qdrant

        Args:
            pdf_path (str): Path to the PDF document to process

        Returns:
            str: Success message confirming storage in Qdrant

        Raises:
            FileNotFoundError: If the specified PDF file doesn't exist
            ValueError: If no documents are loaded or no text chunks are created
            ConnectionError: If connection to Qdrant fails
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")

        # Extract the candidate's name from the file name (without extension)
        candidate_name = os.path.splitext(os.path.basename(pdf_path))[0].replace('_', ' ')

        # Load and preprocess the document
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise ValueError("No documents were loaded from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=250
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")
        
        # Append candidate's name to each chunk
        for split in splits:
            split.page_content = f"Candidate Name is {candidate_name}\n\n{split.page_content}"
            print(f"Chunk Content after edit :\n{split.page_content}\n{'-'*50}")  # Print the chunk content
        # Create and store embeddings in Qdrant
        try:
            qdrant = Qdrant.from_documents(
                splits,
                self.embeddings,
                url=self.qdrant_url,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "✅ Vector DB Successfully Created and Stored in Qdrant!"
    