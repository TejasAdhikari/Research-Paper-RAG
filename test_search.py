from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore


# Interactive search testing
def interactive_search():
    print("🔍 Interactive Search Testing")
    print("Type 'quit' to exit\n")
    
    # Load systems
    generator = EmbeddingGenerator()
    store = VectorStore('research_papers_main')
    
    print(f"📊 Database contains {store.get_stats()['total_documents']} documents\n")
    
    while True:
        query = input("Enter your search query: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            continue
        
        # Search
        query_embedding = generator.embed_single(query)
        results = store.search(query_embedding, n_results=3)
        
        print(f"\n🎯 Results for: '{query}'")
        print("-" * 50)
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            print(f"\n{i+1}. Paper: {metadata['title'][:60]}...")
            print(f"   Section: {metadata['section']} | Type: {metadata['chunk_type']}")
            print(f"   Content: {doc[:200]}...")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    interactive_search()