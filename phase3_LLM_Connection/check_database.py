import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from phase2_semantic_search_system.vector_store import VectorStore

def check_database():
    print("🔍 Checking ChromaDB status...")
    
    # Check if database directory exists
    db_path = Path("chroma_db")
    if not db_path.exists():
        print("❌ ChromaDB directory doesn't exist!")
        return False
    
    try:
        # Connect to database
        store = VectorStore('research_papers_main')
        stats = store.get_stats()
        
        print(f"📊 Database Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Collection name: {stats.get('collection_name', 'Unknown')}")
        
        if stats['total_documents'] == 0:
            print("❌ Database is empty! You need to run Phase 2 first.")
            return False
        else:
            print("✅ Database has documents!")
            
            # Test a sample search
            from phase2_semantic_search_system.embedding_generator import EmbeddingGenerator
            embedding_gen = EmbeddingGenerator()
            
            test_embedding = embedding_gen.embed_single("machine learning")
            results = store.search(test_embedding, n_results=2)
            
            print(f"🔍 Test search results:")
            print(f"  Results type: {type(results)}")
            print(f"  Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
            
            if results and results.get('documents'):
                print(f"  Found {len(results['documents'][0])} documents")
                print(f"  First document preview: {results['documents'][0][0][:100]}...")
            
            return True
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

if __name__ == "__main__":
    check_database()