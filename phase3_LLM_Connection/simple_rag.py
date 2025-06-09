import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))


from phase2_semantic_search_system.embedding_generator import EmbeddingGenerator
from phase2_semantic_search_system.vector_store import VectorStore
from phase3_LLM_Connection.gemini_llm import GeminiLLM
import time


# Simple RAG (Retrieval-Augmented Generation) system
class SimpleRAG:
    def __init__(self):
        print("🔄 Initializing RAG components.")
        
        # Load systems from Phase 2
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore('research_papers_main')
        # Initialize the LLM connection
        self.llm = GeminiLLM()
        
        # 🔍 Debug: Check if database has documents
        stats = self.vector_store.get_stats()
        print(f"📊 Database stats: {stats}")
        
        print("✅ RAG pipeline ready!")
    

    # Answer a question using the RAG system
    def answer_question(self, question, n_results=5):
        print(f"🔍 Question: {question}")
        
        # Step 1: Get relevant documents
        query_embedding = self.embedding_generator.embed_single(question)
        search_results = self.vector_store.search(query_embedding, n_results=n_results)
        
        # # Debug: Print the actual search results structure
        # print("🔍 DEBUG - Search results structure:")
        # print(f"Type: {type(search_results)}")
        # print(f"Keys: {search_results.keys() if isinstance(search_results, dict) else 'Not a dict'}")
        
        # if isinstance(search_results, dict):
        #     for key, value in search_results.items():
        #         print(f"  {key}: {type(value)}")
        #         if isinstance(value, list) and len(value) > 0:
        #             print(f"    First item type: {type(value[0])}")
        #             print(f"    Length: {len(value)}")
        
        # Check if we have documents
        if not search_results or not search_results.get('documents') or not search_results['documents'][0]:
            return "I couldn't find any relevant information to answer your question."
        
        # Step 2: Build context from retrieved documents
        try:
            context = self._build_context(search_results)
            # print(f"🔍 DEBUG - Context built successfully, length: {len(context)}")
        except Exception as e:
            print(f"❌ Error building context: {e}")
            return f"Error building context: {e}"
        
        # Step 3: Create prompt for LLM
        prompt = self._create_prompt(question, context)
        
        # Step 4: Generate answer
        answer = self.llm.generate_response(prompt)
        
        return {
            'answer': answer,
            'sources_found': len(search_results['documents'][0]),
            'context_used': context[:200] + "..." if len(context) > 200 else context
        }
    

    # Build context from search results
    def _build_context(self, search_results):
        contexts = []
        
        # # 🔍 Debug: Check the structure of documents and metadata
        # print("🔍 DEBUG - Building context:")
        # print(f"Documents length: {len(search_results['documents'][0])}")
        # print(f"Metadatas length: {len(search_results.get('metadatas', [[]])[0])}")
        
        # Safely iterate through the documents and their metadata
        documents = search_results['documents'][0]
        metadatas = search_results.get('metadatas', [[]])[0]
        
        for i, doc in enumerate(documents):
            # Safely get metadata
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # print(f"🔍 DEBUG - Document {i}:")
            # print(f"  Document type: {type(doc)}")
            # print(f"  Metadata type: {type(metadata)}")
            # print(f"  Metadata keys: {metadata.keys() if isinstance(metadata, dict) else 'Not a dict'}")
            
            context_piece = f"""
                            Source {i+1}:
                            Paper: {metadata.get('title', 'Unknown') if isinstance(metadata, dict) else 'Unknown'}
                            Section: {metadata.get('section', 'Unknown') if isinstance(metadata, dict) else 'Unknown'}
                            Content: {doc}
                            ---"""
            contexts.append(context_piece)
        
        return "\n".join(contexts)
    

    # Create a prompt for the LLM
    def _create_prompt(self, question, context):
        return f"""You are a research assistant. Answer the question based on the provided context from research papers.

                Context from research papers:
                {context}

                Question: {question}

                Instructions:
                1. Answer based only on the provided context
                2. Cite papers when making specific claims
                3. If the context doesn't fully answer the question, say what information is missing
                4. Be accurate and scholarly in your response

                Answer:"""


# Test the simple RAG system
if __name__ == "__main__":
    try:
        rag = SimpleRAG()
        
        # Test questions
        test_questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "What optimization methods are used in deep learning?"
        ]
        
        for question in test_questions:
            print("\n" + "="*50)
            result = rag.answer_question(question)

            if isinstance(result, dict):
                print(f"Answer: {result['answer']}")
                print(f"Sources used: {result['sources_found']}")
            else:
                print(f"Result: {result}")
            print("-"*50)
    
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("Make sure you've completed Phase 1 & 2 first!")
    