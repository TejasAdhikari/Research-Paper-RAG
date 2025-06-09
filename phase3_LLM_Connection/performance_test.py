import time
from simple_rag import SimpleRAG


# Test RAG system performance
def test_performance():
    print("⚡ Performance Testing")
    print("=" * 30)
    
    rag = SimpleRAG()
    
    test_queries = [
        "What is deep learning?",
        "How do CNNs work?", 
        "What optimization methods are used?",
        "What were the results?",
        "How do transformers compare to RNNs?"
    ]
    
    times = []
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        start_time = time.time()
        try:
            result = rag.answer_question(query)
            end_time = time.time()
            
            query_time = end_time - start_time
            times.append(query_time)
            successful_queries += 1
            
            print(f"  Time: {query_time:.2f}s")
            print(f"  Sources: {result['sources_found']}")
            print(f"  Answer length: {len(result['answer'])} characters")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print("-" * 30)
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📈 Performance Summary:")
        print(f"  Average response time: {avg_time:.2f} seconds")
        print(f"  Successful queries: {successful_queries}/{len(test_queries)}")
        print(f"  Performance rating: {'Excellent' if avg_time < 2 else 'Good' if avg_time < 5 else 'Needs improvement'}")

if __name__ == "__main__":
    test_performance()