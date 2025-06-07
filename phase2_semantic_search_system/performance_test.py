import time
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore


# Test search performance
def test_performance():
    generator = EmbeddingGenerator()
    store = VectorStore('research_papers_main')
    
    test_queries = [
        "machine learning algorithms",
        "neural network training",
        "experimental methodology",
        "optimization techniques",
        "classification accuracy"
    ]
    
    print("⚡ Performance Testing")
    print(f"📊 Database size: {store.get_stats()['total_documents']} documents")
    
    total_time = 0
    
    for query in test_queries:
        start_time = time.time()
        
        # Generate embedding + search
        query_embedding = generator.embed_single(query)
        results = store.search(query_embedding, n_results=5)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        
        print(f"Query: '{query}' | Time: {query_time:.3f}s | Results: {len(results['documents'][0])}")
    
    avg_time = total_time / len(test_queries)
    print(f"\n📈 Average query time: {avg_time:.3f} seconds")
    print(f"🎯 Performance: {'Excellent' if avg_time < 0.1 else 'Good' if avg_time < 0.5 else 'Needs improvement'}")

if __name__ == "__main__":
    test_performance()