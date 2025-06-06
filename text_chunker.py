import json
from pathlib import Path
import logging

# Setup logging.
logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, base_dir="papers", chunk_size=1000, overlap=200):
        self.base_dir = Path(base_dir)
        self.text_dir = self.base_dir / "processed_text"
        self.chunks_dir = self.base_dir / "chunks"
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    # Create chunks from a processed paper.
    def chunk_single_paper(self, paper_file):
        """Create chunks from one processed paper"""
        # Load the paper data from JSON file.
        with open(paper_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        chunks = []
        
        # Always include abstract as separate chunk
        if paper_data.get('abstract'):
            chunks.append({
                'paper_id': paper_data['paper_id'],
                'chunk_type': 'abstract',
                'content': paper_data['abstract'],
                'metadata': {
                    'title': paper_data['title'],
                    'section': 'abstract'
                }
            })
        
        # Chunk the full text
        full_text = paper_data.get('full_text', '')
        if full_text:
            text_chunks = self._create_overlapping_chunks(full_text)
            
            # Create chunks with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'paper_id': paper_data['paper_id'],
                    'chunk_type': 'full_text',
                    'chunk_id': i,
                    'content': chunk_text,
                    'metadata': {
                        'title': paper_data['title'],
                        'section': 'full_text',
                        'chunk_position': i
                    }
                })
        
        logger.info(f"✅ Created {len(chunks)} chunks for {paper_data['paper_id']}")
        return chunks
    
    # Split text into overlapping chunks.
    def _create_overlapping_chunks(self, text):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                if last_period > start + self.chunk_size // 2:
                    chunk_text = text[start:start + last_period + 1]
            
            chunks.append(chunk_text.strip())
            start += self.chunk_size - self.overlap
            
            if start >= len(text):
                break
        
        return chunks
    

# Main entry point.
if __name__ == "__main__":
    chunker = TextChunker()
    
    # Find a processed text file to test
    text_files = list(chunker.text_dir.glob("*.json"))
    
    if text_files:
        chunks = chunker.chunk_single_paper(text_files[0])
        print(f"✅ Created {len(chunks)} chunks")
        print(f"First chunk preview: {chunks[0]['content'][:100]}...")
    else:
        print("❌ No processed text files found - run PDF processing first")