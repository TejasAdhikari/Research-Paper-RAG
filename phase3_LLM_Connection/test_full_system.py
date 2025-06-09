import os
from pathlib import Path


# Check if all prerequisites are met
def check_prerequisites():
    issues = []
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        issues.append("❌ GOOGLE_API_KEY not set")
    
    # Check Phase 2 database
    if not Path("chroma_db").exists():
        issues.append("❌ ChromaDB not found - run Phase 2 first")
    
    # Check chunks
    if not Path("../papers/chunks/all_chunks.json").exists():
        issues.append("❌ Chunks not found - run Phase 1 first")
    
    if issues:
        print("\n".join(issues))
        return False
    
    print("✅ All prerequisites met!")
    return True


# Test the complete RAG pipeline
def test_complete_pipeline():
    if not check_prerequisites():
        return
    
    print("🧪 Testing Complete RAG Pipeline")
    print("=" * 40)
    
    try:
        from simple_rag import SimpleRAG
        
        # Initialize system
        rag = SimpleRAG()
        
        # Test questions with different complexities
        test_cases = [
            {
                'question': "What is machine learning?",
                'expected_keywords': ['learning', 'algorithm', 'data']
            },
            {
                'question': "How do transformers work in deep learning?",
                'expected_keywords': ['attention', 'transformer', 'neural']  
            },
            {
                'question': "What optimization algorithms are commonly used?",
                'expected_keywords': ['optimization', 'gradient', 'learning rate']
            }
        ]
        
        successful_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['question']}")
            
            try:
                result = rag.answer_question(test_case['question'])
                answer = result['answer'].lower()
                
                # Check if answer contains expected keywords
                keywords_found = sum(1 for keyword in test_case['expected_keywords'] 
                                   if keyword.lower() in answer)
                
                print(f"Answer: {result['answer'][:150]}...")
                print(f"Sources: {result['sources_found']}")
                print(f"Keywords found: {keywords_found}/{len(test_case['expected_keywords'])}")
                
                if keywords_found > 0 and result['sources_found'] > 0:
                    print("✅ Test passed")
                    successful_tests += 1
                else:
                    print("⚠️ Test partially successful")
                    
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        # Summary
        print(f"\n📊 Test Summary: {successful_tests}/{len(test_cases)} tests successful")
        
        if successful_tests == len(test_cases):
            print("🎉 RAG system is working perfectly!")
        elif successful_tests > 0:
            print("✅ RAG system is working with minor issues")
        else:
            print("❌ RAG system needs debugging")
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")

if __name__ == "__main__":
    test_complete_pipeline()