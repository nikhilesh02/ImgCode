"""Document processing module for loading and splitting documents"""

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

# Import ImageProcessor for processing images
from .image_processor import ImageProcessor


class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, include_images: bool = True):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            include_images: Whether to process images from data folder
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_images = include_images
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.image_processor = ImageProcessor() if include_images else None
    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a TXT file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFDirectoryLoader(str("data"))
        return loader.load()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, or TXT files

        Args:
            sources: List of URLs, PDF folder paths, or TXT file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            # Check if it's a URL
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue
            
            # Otherwise treat as a file/folder path
            path = Path(src)
            
            # If it's a directory, load all PDFs from it
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            # If it's a .txt file
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            # If it's a PDF file
            elif path.suffix.lower() == ".pdf":
                # Load single PDF
                if path.exists():
                    docs.extend(self.load_from_pdf_dir(path.parent))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, or PDF directory."
                )
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        
        # Also process images from data folder if enabled
        if self.include_images and self.image_processor:
            try:
                image_docs = self.process_images_from_data_folder()
                docs.extend(image_docs)
            except Exception as e:
                print(f"Error processing images: {e}")
        
        return self.split_documents(docs)
    
    def process_images_from_data_folder(self, directory: Union[str, Path] = None) -> List[Document]:
        """
        Process all images from the data folder
        
        Args:
            directory: Path to directory containing images (default: data folder)
            
        Returns:
            List of Documents containing image analyses
        """
        if directory is None:
            directory = Path("data")
        else:
            directory = Path(directory)
        
        if not self.image_processor:
            return []
        
        return self.image_processor.process_images_from_directory(directory)
    
    def get_all_data_files(self, directory: Union[str, Path] = None) -> dict:
        """
        Get all supported files from the data folder (PDFs and images)
        
        Args:
            directory: Path to directory (default: data folder)
            
        Returns:
            Dictionary with 'pdfs' and 'images' lists
        """
        if directory is None:
            directory = Path("data")
        else:
            directory = Path(directory)
        
        result = {"pdfs": [], "images": []}
        
        if not directory.exists():
            return result
        
        # Get PDFs
        pdf_extensions = [".pdf"]
        for ext in pdf_extensions:
            result["pdfs"].extend(list(directory.glob(f"*{ext}")))
            result["pdfs"].extend(list(directory.glob(f"*{ext.upper()}")))
        
        # Get images
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
        for ext in image_extensions:
            result["images"].extend(list(directory.glob(f"*{ext}")))
            result["images"].extend(list(directory.glob(f"*{ext.upper()}")))
        
        return result
