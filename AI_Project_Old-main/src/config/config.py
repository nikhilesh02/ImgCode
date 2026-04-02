"""Configuration module for Agentic RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    LLM_MODEL = "openai:gpt-4o"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Data folder path
    DATA_FOLDER = "data"
    
    # Default URLs - Now using local data folder instead of internet
    DEFAULT_URLS = ["data"]
    
    # Supported file extensions
    SUPPORTED_PDFS = [".pdf"]
    SUPPORTED_IMAGES = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    SUPPORTED_TEXTS = [".txt"]
    
    @classmethod
    def get_data_files(cls):
        """Get all supported files from the data folder"""
        data_path = Path(cls.DATA_FOLDER)
        if not data_path.exists():
            return []
        
        files = []
        for ext in cls.SUPPORTED_PDFS + cls.SUPPORTED_IMAGES + cls.SUPPORTED_TEXTS:
            files.extend(list(data_path.glob(f"*{ext}")))
            files.extend(list(data_path.glob(f"*{ext.upper()}")))
        return sorted(files)
    
    @classmethod
    def get_llm(cls):
        """Initialize and return the LLM model"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        return init_chat_model(cls.LLM_MODEL)
