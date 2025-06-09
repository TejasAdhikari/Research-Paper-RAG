from simple_rag import SimpleRAG
import time

class ChatInterface:
    # Initialize chat interface
    def __init__(self):
        print("🤖 Research Paper Assistant")
        print("=" * 40)
        
        try:
            self.rag = SimpleRAG()
            
            # Get database stats
            stats = self.rag.vector_store.get_stats()
            print(f"📊 Database contains {stats['total_documents']} research paper chunks")
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return
    

    # Start the interactive chat interface
    def start_chat(self):
        print("\n💡 Example questions:")
        print("  - What machine learning algorithms were discussed?")
        print("  - How do transformers work?")  
        print("  - What were the experimental results?")
        print("\nType 'quit' to exit\n")
        
        while True:
            try:
                # Get user input
                user_input = input("🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                # Get response
                print("🤔 Searching papers.")
                start_time = time.time()
                
                result = self.rag.answer_question(user_input)
                
                response_time = time.time() - start_time
                
                # Display response
                print(f"\n🤖 Assistant: {result['answer']}")
                print(f"\n📚 Found {result['sources_found']} relevant sources")
                print(f"⏱️ Response time: {response_time:.2f} seconds\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    chat = ChatInterface()
    if hasattr(chat, 'rag'):  # Only start if initialization succeeded
        chat.start_chat()