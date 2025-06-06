import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import pandas as pd
import logging

# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the ArxivDataCollector class.
class ArxivDataCollector:
    def __init__(self, base_dir="papers"):
        self.base_dir = Path(base_dir)
        self.base_url = "http://export.arxiv.org/api/query"
        self.delay_between_calls = 3  # ArXiv rate limit
        self.setup_directories()
    
    # Create necessary directories.
    def setup_directories(self):
        """Create folder structure"""
        directories = [
            self.base_dir,
            self.base_dir / "raw_pdfs",
            self.base_dir / "processed_text", 
            self.base_dir / "chunks"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directories in {self.base_dir}")

    # Fetch sample papers from ArXiv.
    def fetch_sample_papers(self, category="cs.LG", max_results=10):
        query = f"cat:{category}"
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            # print(response.text)
            papers = self._parse_arxiv_response(response.text)
            logger.info(f"✅ Fetched {len(papers)} papers")
            return papers
        except Exception as e:
            logger.error(f"❌ Error fetching papers: {e}")
            return []

    # Parse the XML response from ArXiv.
    def _parse_arxiv_response(self, xml_content):
        papers = []
        try:
            root = ET.fromstring(xml_content)
            # Define the namespace for Atom XML
            # Atom namespace is required for parsing ArXiv's XML response
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                paper = {
                    'id': entry.find('atom:id', ns).text.split('/')[-1],
                    'title': entry.find('atom:title', ns).text.strip(),
                    'summary': entry.find('atom:summary', ns).text.strip(),
                    'published': entry.find('atom:published', ns).text[:10]
                }
                papers.append(paper)
        except Exception as e:
            logger.error(f"❌ XML parsing error: {e}")
        
        return papers
    

    # Fetch papers and ensure they have PDF URLs.
    def fetch_papers_with_pdfs(self, category, max_results=50):
        papers = self.fetch_sample_papers(category, max_results)
        
        # Add PDF URLs and filter out papers without them
        papers_with_pdfs = []
        for paper in papers:
            # ArXiv PDF URL pattern
            arxiv_id = paper['id']
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            paper['pdf_url'] = pdf_url
            papers_with_pdfs.append(paper)
        
        return papers_with_pdfs



# Test the collector
if __name__ == "__main__":
    collector = ArxivDataCollector()
    print("✅ Directory setup complete!")

    papers = collector.fetch_sample_papers(max_results=5)
    if papers:
        print(f"✅ Successfully fetched {len(papers)} papers")
        print(f"First paper: {papers[0]['title'][:50]}...")
    else:
        print("❌ No papers fetched - check your internet connection")