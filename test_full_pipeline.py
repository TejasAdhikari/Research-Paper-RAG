import json
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore


# Test the complete embedding + storage pipeline with chunks
def test_pipeline_with_chunks():
    # Load chunks
    print("📂 Loading chunks.")
    with open("papers/chunks/all_chunks.json", 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Limit to first 20 for testing
    test_chunks = chunks[:20]
    print(f"🧪 Testing with {len(test_chunks)} chunks")
    
    # Generate embeddings
    print("🧠 Generating embeddings.")
    generator = EmbeddingGenerator()
    texts = [chunk['content'] for chunk in test_chunks]
    embeddings = generator.embed_texts(texts)
    
    # Prepare metadata and IDs for each chunk
    metadatas = []
    ids = []

    for i, chunk in enumerate(test_chunks):
        metadata = {
            'paper_id': str(chunk.get('paper_id', '')),
            'chunk_type': str(chunk.get('chunk_type', '')),
            'title': str(chunk.get('metadata', {}).get('title', ''))[:100]  # Limit length
        }
        metadatas.append(metadata)
        ids.append(f"chunk_{i}")
    
    # Store in vector database
    print("🗄️ Storing in vector database.")
    store = VectorStore("test_papers")
    store.add_documents(texts, embeddings, metadatas, ids)
    
    # Test search
    print("🔍 Testing search.")
    query = "machine learning algorithms"
    query_embedding = generator.embed_single(query)
    results = store.search(query_embedding, n_results=3)
    
    print(f"\n🎯 Search Results for '{query}':")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\nResult {i+1}:")
        print(f"  Title: {metadata['title']}")
        print(f"  Type: {metadata['chunk_type']}")
        print(f"  Content: {doc[:150]}.")
    
    print(f"\n✅ Pipeline test successful!")
    return store, generator

if __name__ == "__main__":
    try:
        store, generator = test_pipeline_with_chunks()
        print("🎉 Ready for full Phase 2 pipeline!")
    except FileNotFoundError:
        print("❌ Chunks file not found. Complete Phase 1 first!")