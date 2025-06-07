import requests
import PyPDF2
from pathlib import Path
import json
import time
import logging

# Setup logging.
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, base_dir="../papers"):
        self.base_dir = Path(base_dir)
        self.pdf_dir = self.base_dir / "raw_pdfs"
        self.text_dir = self.base_dir / "processed_text"
    
    
    # Download and process a PDF file.
    def download_single_pdf(self, pdf_url, paper_id):
        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        
        if pdf_path.exists():
            logger.info(f"📄 PDF already exists: {paper_id}")
            return pdf_path
        
        try:
            # Download the PDF file.
            response = requests.get(pdf_url, timeout=30)
            # Check if the request was successful.
            response.raise_for_status()
            
            # Save the PDF file.
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded: {paper_id}.pdf")
            return pdf_path
            
        except Exception as e:
            logger.error(f"❌ Download failed for {paper_id}: {e}")
            return None
        

    # Extract text from a PDF file.
    def extract_text_from_pdf(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text one page at a time.
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            return None


    # Process a single paper completely.
    def process_single_paper(self, paper_data):
        paper_id = paper_data['id']
        
        # Download PDF
        pdf_path = self.download_single_pdf(paper_data['pdf_url'], paper_id)
        if not pdf_path:
            return None
        
        # Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        if not full_text:
            return None
        
        # Structure the data
        structured_data = {
            'paper_id': paper_id,
            'title': paper_data['title'],
            'abstract': paper_data['summary'],
            'full_text': full_text,
            'metadata': paper_data
        }
        
        # Save processed text
        text_path = self.text_dir / f"{paper_id}.json"
        with open(text_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Processed: {paper_id}")
        return structured_data
    

if __name__ == "__main__":
    # Sample paper with PDF URL
    sample_paper = {
        'id': '2505.23765',  # sample pdf actual ID
        'title': 'Sample Paper',
        'summary': 'Sample abstract',
        'pdf_url': 'https://arxiv.org/pdf/2505.23765.pdf'  # sample pdf actual URL
    }
    
    processor = PDFProcessor()
    result = processor.process_single_paper(sample_paper)
    
    if result:
        print("✅ PDF processing successful!")
        print(f"Text length: {len(result['full_text'])} characters")
    else:
        print("❌ PDF processing failed")