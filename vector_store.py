import chromadb
from pathlib import Path
import numpy as np
import logging


# Set up logging
logger = logging.getLogger(__name__)


class VectorStore:
    # Initialize ChromaDB vector store
    def __init__(self, collection_name="research_papers"):
        # Create persistent database
        db_path = Path("chroma_db")
        db_path.mkdir(exist_ok=True)
        
        print(f"🔄 Initializing ChromaDB at {db_path}")
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(collection_name)
        
        current_count = self.collection.count()
        print(f"✅ Vector store ready! Current documents: {current_count}")
    

    # Add documents with embeddings to the store
    def add_documents(self, texts, embeddings, metadatas, ids):
        print(f"🔄 Adding {len(texts)} documents...")
        
        # Convert numpy array to list for ChromaDB
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(texts)} documents. Total: {self.collection.count()}")
    

    # Search for similar documents using an embedding
    def search(self, query_embedding, n_results=5):
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    

    # Get database statistics
    def get_stats(self):
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }


# Test the vector store
if __name__ == "__main__":
    # Test with dummy data
    store = VectorStore("test_collection")
    
    # Create some test data
    texts = ["Hello world", "Machine learning", "Vector database"]
    embeddings = np.random.rand(3, 384)  # 384 is MiniLM dimension
    metadatas = [{"source": f"test_{i}"} for i in range(3)]
    ids = [f"doc_{i}" for i in range(3)]
    
    # Add to store
    store.add_documents(texts, embeddings, metadatas, ids)
    
    # Test search
    query_embedding = np.random.rand(384)
    results = store.search(query_embedding, n_results=2)
    
    print(f"✅ Search test successful! Found {len(results['documents'][0])} results")