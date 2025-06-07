try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    import numpy as np
    print("✅ All Phase 2 libraries installed successfully!")
    
    # Quick test
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")