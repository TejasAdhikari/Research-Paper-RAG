from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize with a sentence transformer model
        print(f"🔄 Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")
    
    
    # Convert list of texts to embeddings
    def embed_texts(self, texts, batch_size=32):
        print(f"🔄 Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True  # Important for similarity search
        )
        
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        return embeddings
    

    # Embed a single text (for queries)
    def embed_single(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0]


# Test the embedding generator
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    
    # Test with sample texts
    sample_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by biological neurons",
        "Deep learning uses multiple layers of neural networks"
    ]
    
    embeddings = generator.embed_texts(sample_texts)
    print(f"✅ Test successful! Generated {embeddings.shape[0]} embeddings")
    
    # Test single embedding
    query_embedding = generator.embed_single("What is AI?")
    print(f"✅ Single embedding shape: {query_embedding.shape}")