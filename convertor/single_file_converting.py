# Converting Single File

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def pdf_to_text_via_ocr(pdf_path, output_dir):
    # Extract file name without extension
    file_name = os.path.basename(pdf_path).split('.')[0]
    
    # Set output paths for text and Word file
    txt_file_path = os.path.join(output_dir, f"{file_name}.txt")
    docx_file_path = os.path.join(output_dir, f"{file_name}.docx")
    
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    extracted_text = ""
    
    try:
        for page_num in range(pdf_document.page_count):
            # Get page and render it as an image
            page = pdf_document[page_num]
            pix = page.get_pixmap()  # render page as image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR on the image
            text = pytesseract.image_to_string(img)
            extracted_text += f"Page {page_num + 1}:\n{text}\n\n"
        
        # Write the extracted text to a .txt file
        with open(txt_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        print(f"Text file saved at: {txt_file_path}")
        
        # Write to a Word document if needed
        from docx import Document
        doc = Document()
        doc.add_paragraph(extracted_text)
        doc.save(docx_file_path)
        print(f"Word file saved at: {docx_file_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        pdf_document.close()

# Specify paths
pdf_file_path = r"D:\Project\Dataset\pdfs\2022080523.pdf"
output_directory = os.path.dirname(pdf_file_path)

# Run OCR on the PDF
pdf_to_text_via_ocr(pdf_file_path, output_directory)
