"""Image processing module using OpenAI Vision API"""

import base64
import os
from pathlib import Path
from typing import List, Union, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document


class ImageProcessor:
    """Handles image processing using OpenAI Vision API"""
    
    def __init__(self, model: str = "gpt-4o", detail: str = "high"):
        """
        Initialize image processor
        
        Args:
            model: OpenAI model to use for vision (default: gpt-4o)
            detail: Detail level for image processing - "low", "high", or "auto"
        """
        self.model = model
        self.detail = detail
        self.llm = ChatOpenAI(model=model, temperature=0)
    
    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def process_image(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> str:
        """
        Process a single image using OpenAI Vision API
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt to guide the vision analysis (optional)
            
        Returns:
            Description/analysis of the image
        """
        if prompt is None:
            prompt = """Analyze this image in detail. Describe:
            1. What is shown in the image
            2. Any text or labels visible
            3. Key objects, people, or elements
            4. Any patterns or important features
            5. The overall context and meaning of the image"""
        
        # Encode the image
        base64_image = self.encode_image(image_path)
        
        # Create message with image
        from langchain_core.messages import HumanMessage
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": self.detail
                    },
                },
            ]
        )
        
        # Get response from LLM
        response = self.llm.invoke([message])
        return response.content
    
    def process_image_to_document(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> Document:
        """
        Process an image and return a Document with the analysis
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt to guide the vision analysis (optional)
            
        Returns:
            Document containing the image analysis
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Process the image
        analysis = self.process_image(image_path, prompt)
        
        # Create a Document
        doc = Document(
            page_content=analysis,
            metadata={
                "source": str(image_path),
                "source_type": "image",
                "file_name": image_path.name,
                "file_size": image_path.stat().st_size,
            }
        )
        
        return doc
    
    def process_images_from_directory(self, directory: Union[str, Path], extensions: List[str] = None, prompt: Optional[str] = None) -> List[Document]:
        """
        Process all images in a directory
        
        Args:
            directory: Path to directory containing images
            extensions: List of file extensions to process (default: common image formats)
            prompt: Custom prompt to guide the vision analysis
            
        Returns:
            List of Documents containing image analyses
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
        
        directory = Path(directory)
        documents = []
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Process each image
        for image_path in sorted(image_files):
            try:
                doc = self.process_image_to_document(image_path, prompt)
                documents.append(doc)
                print(f"Processed: {image_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
        
        return documents
    
    def process_uploaded_image(self, uploaded_file, prompt: Optional[str] = None) -> str:
        """
        Process an image uploaded via Streamlit
        
        Args:
            uploaded_file: Streamlit uploaded file object
            prompt: Custom prompt to guide the vision analysis
            
        Returns:
            Description/analysis of the image
        """
        if prompt is None:
            prompt = """Analyze this image in detail. Describe:
            1. What is shown in the image
            2. Any text or labels visible
            3. Key objects, people, or elements
            4. Any patterns or important features
            5. The overall context and meaning of the image"""
        
        # Read the uploaded file
        import streamlit as st
        from langchain_core.messages import HumanMessage
        
        # Get file bytes and encode to base64
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode("utf-8")
        
        # Determine image format
        file_extension = uploaded_file.name.split(".")[-1].lower()
        mime_type = f"image/{file_extension}"
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        
        # Create message with image
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": self.detail
                    },
                },
            ]
        )
        
        # Get response from LLM
        response = self.llm.invoke([message])
        return response.content
