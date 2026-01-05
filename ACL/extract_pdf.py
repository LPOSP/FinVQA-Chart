
import sys
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("No PDF library found")
        sys.exit(1)

file_path = "C:\\Users\\Dell\\Desktop\\ACL\\ACL_Dataset_VLM (2).pdf"

try:
    reader = PdfReader(file_path)
    text = ""
    # Read pages 5 to 15
    for i in range(5, min(15, len(reader.pages))):
        page = reader.pages[i]
        text += f"--- Page {i} ---\n"
        text += page.extract_text() + "\n"
    print(f"Text length: {len(text)}")
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
