import logging
from arxiv_collector import ArxivDataCollector
from pdf_processor import PDFProcessor  
from text_chunker import TextChunker
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Phase 1: Data Collection Pipeline
def run_phase1_pipeline():    
    print("🚀 Starting Phase 1: Data Collection Pipeline")
    print("=" * 50)
    
    # Step 1: Collect paper metadata
    print("\n📊 Step 1: Collecting papers from ArXiv")
    collector = ArxivDataCollector()
    # print(collector)
    
    # Just one category for testing
    categories = ['cs.LG']  # Just Machine Learning
    papers = []
    
    # Collect papers with PDFs
    for category in categories:
        category_papers = collector.fetch_papers_with_pdfs(category, max_results=20)
        papers.extend(category_papers)
    
    logger.info(f"✅ Collected {len(papers)} papers with PDF URLs")
    
    # Save metadata
    metadata_file = collector.base_dir / "collected_papers.json"
    with open(metadata_file, 'w') as f:
        json.dump(papers, f, indent=2)
    
    # Step 2: Process PDFs (start with just 5 papers)
    print(f"\n📄 Step 2: Processing first 5 PDFs")
    processor = PDFProcessor()
    
    processed_papers = []
    for i, paper in enumerate(papers[:5]):  # Limit to 5 for testing
        print(f"Processing paper {i+1}/5: {paper['title'][:50]}...")
        result = processor.process_single_paper(paper)
        if result:
            processed_papers.append(result)
    
    logger.info(f"✅ Successfully processed {len(processed_papers)} papers")
    
    # Step 3: Create chunks
    print(f"\n✂️ Step 3: Creating text chunks")
    chunker = TextChunker()
    
    all_chunks = []
    text_files = list(processor.text_dir.glob("*.json"))
    
    for text_file in text_files:
        chunks = chunker.chunk_single_paper(text_file)
        all_chunks.extend(chunks)
    
    # Save all chunks
    chunks_file = chunker.chunks_dir / "all_chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Created {len(all_chunks)} total chunks")
    
    # Summary
    print(f"""
    🎉 Phase 1 Complete!
    
    📊 Results:
    - Papers collected: {len(papers)}
    - Papers processed: {len(processed_papers)} 
    - Text chunks created: {len(all_chunks)}
    
    📁 Files created:
    - ../papers/collected_papers.json
    - ../papers/processed_text/*.json
    - ../papers/chunks/all_chunks.json
    
    🚀 Ready for Phase 2: Vector Embeddings!
    """)

if __name__ == "__main__":
    run_phase1_pipeline()