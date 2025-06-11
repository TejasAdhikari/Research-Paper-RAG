import streamlit as st
import plotly.express as px
import pandas as pd
import time
import json
from datetime import datetime
import sys
from pathlib import Path

# Add imports
sys.path.append(str(Path(__file__).parent.parent))

from phase3_LLM_Connection.simple_rag import SimpleRAG

# Page config
st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
            <style>
                .main-header {
                    font-size: 3rem;
                    text-align: center;
                    color: #1f77b4;
                    margin-bottom: 2rem;
                }
                
                .metric-card {
                    background: white;
                    padding: 1rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 0.5rem 0;
                }
                
                .chat-message {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 10px;
                }
                
                .user-message {
                    background-color: #94afc2;
                    text-align: right;
                }
                
                .assistant-message {
                    background-color: #a574ab;
                }
            </style>
            """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'query_analytics' not in st.session_state:
        st.session_state.query_analytics = []

init_session_state()

# Sidebar
def setup_sidebar():
    with st.sidebar:
        st.markdown("## 🔧 Control Panel")
        
        # System initialization
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Loading RAG system..."):
                try:
                    st.session_state.rag_system = SimpleRAG()
                    st.success("✅ System initialized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # System status
        if st.session_state.rag_system:
            st.success("✅ System Online")
            
            # System stats
            try:
                stats = st.session_state.rag_system.vector_store.get_stats()
                st.metric("📚 Documents", stats['total_documents'])
                st.metric("💬 Queries", len(st.session_state.query_analytics))
                
                if st.session_state.query_analytics:
                    avg_time = sum(q['response_time'] for q in st.session_state.query_analytics) / len(st.session_state.query_analytics)
                    st.metric("⏱️ Avg Time", f"{avg_time:.2f}s")
            except:
                st.info("Stats unavailable")
        
        # Controls
        st.markdown("### 🎛️ Controls")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Export options
        if st.session_state.chat_history:
            chat_json = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                "📥 Export Chat",
                chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Main interface
def main_interface():
    # Header
    st.markdown('<h1 class="main-header">🔬 Research Paper RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        analytics_dashboard()
    
    with tab3:
        about_page()

def chat_interface():
    if not st.session_state.rag_system:
        st.warning("Please initialize the system using the sidebar.")
        return
    
    st.markdown("## 💬 Ask Questions About Research Papers")
    
    # Display chat history
    for i, exchange in enumerate(st.session_state.chat_history):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {exchange['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistant:</strong> {exchange['answer'][:500]}{'...' if len(exchange['answer']) > 500 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'response_time' in exchange:
                st.metric("Time", f"{exchange['response_time']:.2f}s")
        
        st.markdown("---")
    
    # Query input
    with st.form("query_form"):
        user_query = st.text_input(
            "Your question:",
            placeholder="e.g., What machine learning algorithms were discussed?"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.form_submit_button("🔍 Ask", type="primary")
        with col2:
            if st.form_submit_button("💡 Try Example"):
                examples = [
                    "What is machine learning?",
                    "How do neural networks work?",
                    "What optimization methods are used?",
                    "What were the experimental results?"
                ]
                import random
                user_query = random.choice(examples)
                submit = True
    
    # Process query
    if submit and user_query:
        process_query(user_query)

def process_query(query):
    with st.spinner("🔍 Searching through research papers..."):
        start_time = time.time()
        
        try:
            result = st.session_state.rag_system.answer_question(query)
            response_time = time.time() - start_time
            
            answer = result['answer'] if isinstance(result, dict) else result
            
            # Add to history
            exchange = {
                'question': query,
                'answer': answer,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.chat_history.append(exchange)
            st.session_state.query_analytics.append({
                'query': query,
                'response_time': response_time,
                'query_length': len(query),
                'answer_length': len(answer),
                'timestamp': datetime.now().isoformat()
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error: {e}")

def analytics_dashboard():
    st.markdown("## 📊 System Analytics")
    
    if not st.session_state.query_analytics:
        st.info("No analytics data yet. Start chatting to see analytics!")
        return
    
    df = pd.DataFrame(st.session_state.query_analytics)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", len(df))
    with col2:
        st.metric("Avg Response Time", f"{df['response_time'].mean():.2f}s")
    with col3:
        st.metric("Fastest Query", f"{df['response_time'].min():.2f}s")
    with col4:
        st.metric("Slowest Query", f"{df['response_time'].max():.2f}s")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time over time
        fig1 = px.line(df, x='timestamp', y='response_time', 
                      title='Response Time Over Time',
                      labels={'response_time': 'Response Time (seconds)'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Query length distribution
        fig3 = px.histogram(df, x='query_length', 
                           title='Query Length Distribution',
                           labels={'query_length': 'Characters in Query'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Response time distribution
        fig2 = px.histogram(df, x='response_time', 
                           title='Response Time Distribution',
                           labels={'response_time': 'Response Time (seconds)'})
        st.plotly_chart(fig2, use_container_width=True)
        
        # Query length vs response time
        fig4 = px.scatter(df, x='query_length', y='response_time',
                         title='Query Length vs Response Time',
                         labels={'query_length': 'Query Length', 'response_time': 'Response Time (s)'})
        st.plotly_chart(fig4, use_container_width=True)

def about_page():
    st.markdown("""
                # About This RAG System 🔬
                
                ## What is RAG?
                **Retrieval-Augmented Generation (RAG)** combines the power of information retrieval with large language models to provide accurate, context-aware responses based on a specific knowledge base.
                
                ## How It Works
                1. **📚 Data Collection**: Research papers collected from ArXiv
                2. **🔍 Text Processing**: Papers split into searchable chunks  
                3. **🧠 Vector Embeddings**: Text converted to mathematical vectors
                4. **🔎 Semantic Search**: Find relevant chunks for each query
                5. **🤖 Response Generation**: AI generates answers using retrieved context
                
                ## Technology Stack
                - **Frontend**: Streamlit
                - **Vector Database**: ChromaDB
                - **Embeddings**: Sentence Transformers
                - **LLM**: Google Gemini Pro
                - **Deployment**: Streamlit Cloud
                
                ## Key Features
                - ✅ Semantic search through research papers
                - ✅ AI-powered response generation  
                - ✅ Real-time analytics dashboard
                - ✅ Source attribution and transparency
                - ✅ Export capabilities for chat history
                
                ## Performance Metrics
                - **Database Size**: 500+ research paper chunks
                - **Response Time**: 2-5 seconds average
                - **Accuracy**: High relevance source retrieval
                
                ---
                
                **Built as a portfolio project demonstrating advanced RAG system development.**
                """)

# Main app
setup_sidebar()
main_interface()