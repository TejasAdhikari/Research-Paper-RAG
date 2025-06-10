import streamlit as st
import sys
from pathlib import Path


# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from phase3_LLM_Connection.simple_rag import SimpleRAG

# Page config
st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🔬 Research Paper RAG Assistant")
st.markdown("Ask questions about research papers and get AI-powered answers!")

# Sidebar for system controls
with st.sidebar:
    st.header("🔧 System Controls")
    
    if st.button("🚀 Initialize RAG System"):
        with st.spinner("Loading RAG system..."):
            try:
                st.session_state.rag_system = SimpleRAG()
                st.success("✅ RAG system initialized!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    if st.session_state.rag_system:
        st.success("✅ System Ready!")
        
        # Show system stats
        try:
            stats = st.session_state.rag_system.vector_store.get_stats()
            st.metric("Documents in Database", stats['total_documents'])
        except:
            st.info("System stats unavailable")

# Main interface
if not st.session_state.rag_system:
    st.warning("Please initialize the RAG system using the sidebar.")
    st.markdown("""
                ### How to get started:
                1. Click "🚀 Initialize RAG System" in the sidebar
                2. Wait for the system to load (this may take a moment)
                3. Start asking questions about research papers!
                
                ### Example questions:
                - "What is machine learning?"
                - "How do neural networks work?"
                - "What optimization algorithms are used?"
                """)
else:
    # Chat interface
    st.markdown("## 💬 Chat Interface")
    
    # Display chat history
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange['question'])
        
        with st.chat_message("assistant"):
            st.write(exchange['answer'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about research papers."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_system.answer_question(prompt)
                    answer = result['answer'] if isinstance(result, dict) else result
                    st.write(answer)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        'question': prompt,
                        'answer': answer
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")