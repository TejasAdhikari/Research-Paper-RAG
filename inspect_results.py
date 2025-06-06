import json
from pathlib import Path

def inspect_phase1_results():
    base_dir = Path("papers")
    
    # Check collected papers
    if (base_dir / "collected_papers.json").exists():
        with open(base_dir / "collected_papers.json", encoding='utf-8') as f:
            papers = json.load(f)
        print(f"📊 Collected Papers: {len(papers)}")
        print(f"Sample title: {papers[0]['title']}")
    
    # Check processed text
    text_files = list((base_dir / "processed_text").glob("*.json"))
    print(f"📄 Processed Papers: {len(text_files)}")
    
    # Check chunks
    if (base_dir / "chunks" / "all_chunks.json").exists():
        with open(base_dir / "chunks" / "all_chunks.json", encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✂️ Text Chunks: {len(chunks)}")
        print(f"Sample chunk: {chunks[0]['content'][:100]}")

if __name__ == "__main__":
    inspect_phase1_results()