"""Test script to verify image processing functionality"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.document_ingestion.document_processor import DocumentProcessor


def test_image_processing():
    """Test image processing with OpenAI Vision"""
    print("🧪 Testing Image Processing...\n")
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Test 1: Process image files from data/ folder
    data_folder = Path("data/")
    
    if data_folder.exists():
        print(f"📁 Found data folder: {data_folder}")
        print(f"📂 Contents: {list(data_folder.rglob('*'))}")
        
        # Process all supported files in the folder
        documents = processor.process([str(data_folder)])
        
        print(f"\n✅ Processed {len(documents)} document chunks")
        
        if documents:
            print("\n📄 First document preview:")
            print("-" * 50)
            print(documents[0].page_content[:300] + "...")
            print("-" * 50)
            print(f"\n📋 Source: {documents[0].metadata.get('source', 'Unknown')}")
            print(f"📋 Type: {documents[0].metadata.get('type', 'Unknown')}")
    else:
        print("⚠️  data/ folder not found. Creating sample folder structure...")
        data_folder.mkdir(exist_ok=True)
        print(f"✅ Created {data_folder}")
        print("\nTo test image processing:")
        print("1. Place image files (.png, .jpg, .jpeg) in the data/ folder")
        print("2. Run this script again")


if __name__ == "__main__":
    test_image_processing()
