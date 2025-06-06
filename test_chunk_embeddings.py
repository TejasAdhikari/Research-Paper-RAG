import json
from embedding_generator import EmbeddingGenerator

# Load a few chunks to test
chunks_file = "papers/chunks/all_chunks.json"

try:
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"📂 Loaded {len(chunks)} chunks")
    
    # Test with first 5 chunks
    test_chunks = chunks[:5]
    test_texts = [chunk['content'] for chunk in test_chunks]
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_texts(test_texts)
    
    print(f"✅ Successfully embedded {len(test_chunks)} chunks!")
    print(f"📊 Embedding shape: {embeddings.shape}")
    
except FileNotFoundError:
    print("❌ Chunks file not found. Make sure Phase 1 is complete!")