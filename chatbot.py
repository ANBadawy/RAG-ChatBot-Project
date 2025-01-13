# # chatbot.py

# import os
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.vectorstores import Qdrant
# from langchain_ollama import ChatOllama
# from qdrant_client import QdrantClient
# from langchain import PromptTemplate
# from langchain.chains import RetrievalQA
# import streamlit as st

# class ChatbotManager:
#     def __init__(
#         self,
#         model_name: str = "BAAI/bge-small-en",
#         device: str = "cpu",
#         encode_kwargs: dict = {"normalize_embeddings": True},
#         llm_model: str = "llama3.2:3b",
#         llm_temperature: float = 0.7,
#         qdrant_url: str = "http://localhost:6333",
#         collection_name: str = "vector_db",
#     ):
#         """
#         Initializes the ChatbotManager with embedding models, LLM, and vector store.

#         Args:
#             model_name (str): The HuggingFace model name for embeddings.
#             device (str): The device to run the model on ('cpu' or 'cuda').
#             encode_kwargs (dict): Additional keyword arguments for encoding.
#             llm_model (str): The local LLM model name for ChatOllama.
#             llm_temperature (float): Temperature setting for the LLM.
#             qdrant_url (str): The URL for the Qdrant instance.
#             collection_name (str): The name of the Qdrant collection.
#         """
#         self.model_name = model_name
#         self.device = device
#         self.encode_kwargs = encode_kwargs
#         self.llm_model = llm_model
#         self.llm_temperature = llm_temperature
#         self.qdrant_url = qdrant_url
#         self.collection_name = collection_name

#         # Initialize Embeddings
#         self.embeddings = HuggingFaceBgeEmbeddings(
#             model_name=self.model_name,
#             model_kwargs={"device": self.device},
#             encode_kwargs=self.encode_kwargs,
#         )

#         # Initialize Local LLM
#         self.llm = ChatOllama(
#             model=self.llm_model,
#             temperature=self.llm_temperature,
#             # Add other parameters if needed
#         )

#         # Define the prompt template
#         self.prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer. Answer must be detailed and well explained.
# Helpful answer:
# """

#         # Initialize Qdrant client
#         self.client = QdrantClient(
#             url=self.qdrant_url, prefer_grpc=False
#         )

#         # Initialize the Qdrant vector store
#         self.db = Qdrant(
#             client=self.client,
#             embeddings=self.embeddings,
#             collection_name=self.collection_name
#         )

#         # Initialize the prompt
#         self.prompt = PromptTemplate(
#             template=self.prompt_template,
#             input_variables=['context', 'question']
#         )

#         # Initialize the retriever
#         self.retriever = self.db.as_retriever(search_kwargs={"k": 5})

#         # Define chain type kwargs
#         self.chain_type_kwargs = {"prompt": self.prompt}

#         # Initialize the RetrievalQA chain with return_source_documents=False
#         self.qa = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=self.retriever,
#             return_source_documents=False,  # Set to False to return only 'result'
#             chain_type_kwargs=self.chain_type_kwargs,
#             verbose=False
#         )


#     def get_response(self, query: str) -> str:
#         """
#         Processes the user's query and returns the chatbot's response.

#         Args:
#             query (str): The user's input question.

#         Returns:
#             str: The chatbot's response.
#         """
#         try:
#             response = self.qa.run(query)
#             return response
#         except Exception as e:
#             raise RuntimeError(f"Error processing request: {e}")


##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################




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
        Initializes the ChatbotManager optimized for CV analysis and talent acquisition.
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

        # Initialize Local LLM with optimized parameters
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
            num_ctx=4096,  # Increased context window
            top_k=50,      # More diverse token sampling
            top_p=0.9,     # Slightly more focused sampling
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
            url=self.qdrant_url, 
            prefer_grpc=False,
            timeout=10.0  # Increased timeout for larger documents
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

        # Enhanced retriever with corrected search parameters
        self.retriever = self.db.as_retriever(
            search_kwargs={
                "k": 8  # Increased number of relevant chunks for better context
            }
        )

        # Chain type kwargs with enhanced settings
        self.chain_type_kwargs = {
            "prompt": self.prompt,
            "verbose": False  # Set to True for debugging if needed
        }

        # Initialize the RetrievalQA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        """
        Processes the CV-related query and returns an expert HR analysis.
        
        Args:
            query (str): The user's input question about CVs or candidates.
        
        Returns:
            str: Detailed professional analysis based on CV content.
        """
        try:
            # Enhance the query for better CV-specific retrieval
            enhanced_query = self._enhance_query(query)
            
            # Get the response
            response = self.qa.run(enhanced_query)
            
            # Format the response for better readability
            formatted_response = self._format_response(response)
            
            return formatted_response

        except Exception as e:
            error_msg = f"Error analyzing CVs: {str(e)}"
            st.error(error_msg)
            return "I apologize, but I encountered an error while analyzing the CVs. Please try rephrasing your question or contact support if the issue persists."

    def _enhance_query(self, query: str) -> str:
        """
        Enhances the query for better CV-specific retrieval.
        """
        # Add HR-specific context to the query
        hr_prefixes = {
            "skills": "Regarding professional skills and competencies,",
            "experience": "In terms of work experience and background,",
            "education": "Concerning educational qualifications,",
            "comparison": "Comparing the candidates' profiles,",
        }
        
        # Determine query type and add appropriate prefix
        for key, prefix in hr_prefixes.items():
            if key in query.lower():
                return f"{prefix} {query}"
                
        return query

    def _format_response(self, response: str) -> str:
        """
        Formats the response for better readability.
        """
        # Remove redundant whitespace
        response = " ".join(response.split())
        
        # Add paragraph breaks for readability
        response = response.replace(". ", ".\n\n")

        
        return response.strip()