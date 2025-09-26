#!/usr/bin/env python3
"""
Test script for image processing functionality
Creates a simple test image with text and tests OCR extraction
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
import io

# Add backend to path
sys.path.append('backend')

def create_test_image():
    """Create a simple test image with text"""
    # Create a white image
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Add some test text
    test_text = """StarRAG Bot Image Processing Test
    
This is a test image to verify OCR functionality.
The system should extract this text successfully.

Key Features:
• Document processing
• Image text extraction  
• Multi-format support
• Offline processing

Contact: test@example.com
Phone: (555) 123-4567"""
    
    # Draw the text
    draw.multiline_text((50, 50), test_text, fill='black', font=font, spacing=10)
    
    # Save the test image
    test_image_path = 'test_image.png'
    img.save(test_image_path)
    print(f"✅ Created test image: {test_image_path}")
    return test_image_path

def test_ocr_extraction():
    """Test OCR text extraction"""
    try:
        from app import RAGPipeline
        
        # Create test image
        test_image_path = create_test_image()
        
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Test OCR extraction
        print("\n🔍 Testing OCR extraction...")
        extracted_text = rag.extract_text_from_image(test_image_path)
        
        print("\n📝 Extracted Text:")
        print("-" * 50)
        print(extracted_text)
        print("-" * 50)
        
        # Check if key phrases were extracted
        key_phrases = [
            "StarRAG Bot",
            "Image Processing", 
            "OCR functionality",
            "test@example.com",
            "(555) 123-4567"
        ]
        
        found_phrases = []
        for phrase in key_phrases:
            if phrase.lower() in extracted_text.lower():
                found_phrases.append(phrase)
                print(f"✅ Found: {phrase}")
            else:
                print(f"❌ Missing: {phrase}")
        
        # Calculate success rate
        success_rate = len(found_phrases) / len(key_phrases) * 100
        print(f"\n📊 OCR Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 60:
            print("🎉 OCR test PASSED! Image processing is working correctly.")
        else:
            print("⚠️  OCR test partially successful. Some text may not be extracted accurately.")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🧹 Cleaned up test image: {test_image_path}")
        
        return success_rate >= 60
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Error during OCR test: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("🔍 Testing dependencies...")
    
    dependencies = [
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('easyocr', 'easyocr'),
        ('pytesseract', 'pytesseract')
    ]
    
    missing_deps = []
    
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install " + " ".join(missing_deps))
        return False
    
    # Test Tesseract specifically
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract OCR version: {version}")
    except Exception as e:
        print(f"❌ Tesseract OCR not properly configured: {e}")
        print("Please install Tesseract OCR and add it to your PATH")
        return False
    
    return True

def main():
    print("🚀 Testing StarRAG Bot Image Processing")
    print("=" * 50)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n❌ Dependency test failed. Please install missing components.")
        return False
    
    print("\n✅ All dependencies are installed!")
    
    # Test OCR functionality
    if test_ocr_extraction():
        print("\n🎉 Image processing test completed successfully!")
        print("Your RAG bot is ready to process images!")
        return True
    else:
        print("\n❌ Image processing test failed.")
        print("Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
