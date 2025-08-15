from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    # If PDF
    if ext == '.pdf':
        pages = convert_from_path(file_path, 300, poppler_path=r"C:\poppler-23.08.0\bin")
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
    
    # If image (png, jpg, jpeg)
    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
    
    else:
        raise ValueError("Unsupported file format. Use PDF, PNG, or JPG.")
