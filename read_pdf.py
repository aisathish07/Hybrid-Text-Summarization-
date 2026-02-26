import PyPDF2
import sys

pdf_path = sys.argv[1]

with open(pdf_path, 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    
    print(f"Total pages: {len(pdf_reader.pages)}\n")
    print("="*80)
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        print(f"\n--- Page {page_num} ---\n")
        text = page.extract_text()
        print(text)
        print("\n" + "="*80)
