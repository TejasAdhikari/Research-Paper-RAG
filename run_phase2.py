import json
import time
from pathlib import Path
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore


# Run the complete Phase 2 pipeline
def run_phase2():
    print("🚀 Phase 2: Vector Database Setup")
    print("=" * 50)
    
    # Load all chunks from Phase 1
    chunks_file = "papers/chunks/all_chunks.json"
    print(f"📂 Loading chunks from {chunks_file}.")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✅ Loaded {len(chunks)} chunks")
    except FileNotFoundError:
        print("❌ Chunks file not found! Make sure Phase 1 is complete.")
        return
    
    # Initialize systems
    print("🧠 Initializing embedding generator.")
    generator = EmbeddingGenerator('all-MiniLM-L6-v2')
    
    print("🗄️ Initializing vector store.")
    store = VectorStore('research_papers_main')
    
    # Check if already processed
    current_count = store.get_stats()['total_documents']
    if current_count >= len(chunks):
        print(f"✅ Database already contains {current_count} documents. Skipping embedding generation.")
    else:
        # Generate embeddings for all chunks
        print(f"⚡ Generating embeddings for {len(chunks)} chunks.")
        start_time = time.time()
        
        texts = [chunk['content'] for chunk in chunks]
        embeddings = generator.embed_texts(texts, batch_size=32)
        
        end_time = time.time()
        print(f"✅ Embeddings generated in {end_time - start_time:.2f} seconds")
        
        # Prepare metadata and IDs for each chunk
        print("📝 Preparing metadata.")
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                'paper_id': str(chunk.get('paper_id', 'unknown')),
                'chunk_type': str(chunk.get('chunk_type', 'text')),
                'title': str(chunk.get('metadata', {}).get('title', 'Unknown'))[:200],
                'section': str(chunk.get('metadata', {}).get('section', 'unknown'))
            }
            metadatas.append(metadata)
            ids.append(f"chunk_{i}_{chunk.get('paper_id', 'unknown')}")
        
        # Add to vector store
        print("💾 Adding to vector database.")
        store.add_documents(texts, embeddings, metadatas, ids)
    
    # Test the retrieval system
    print("\n🧪 Testing retrieval system.")
    test_queries = [
        "What is machine learning?",
        "neural network architecture", 
        "experimental results",
        "optimization algorithm"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        query_embedding = generator.embed_single(query)
        results = store.search(query_embedding, n_results=2)
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"  {i+1}. [{metadata['chunk_type']}] {doc[:80]}.")
    
    # Final statistics
    stats = store.get_stats()
    print(f"""
    🎉 Phase 2 Complete!
    
    📊 Final Statistics:
    - Total documents in database: {stats['total_documents']}
    - Vector database location: ./chroma_db/
    - Embedding model: all-MiniLM-L6-v2 (384 dimensions)
    
    ✅ Your semantic search system is ready!
    🚀 Next: Phase 3 - LLM Integration with Gemini
    """)
    
    return store, generator

if __name__ == "__main__":
    store, generator = run_phase2()