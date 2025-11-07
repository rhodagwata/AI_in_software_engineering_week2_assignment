"""
Intelligent Document Analysis System
A comprehensive system for querying engineering reports with content + metadata
Features:
- Document content search
- Metadata filtering and queries
- Project tracking
- Timeline analysis
- Challenge/lesson extraction
- Multi-modal querying
"""

import streamlit as st
import os
import boto3
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    SimpleDirectoryReader,
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor, MetadataReplacementPostProcessor
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.response_synthesizers import ResponseMode


# ==================== CONFIGURATION ====================
class Config:
    """Configuration settings"""
    # AWS Configuration
    AWS_REGION = "us-east-1"
    CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    # Embedding Model
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    
    # Chunking Configuration
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 100
    
    # Storage
    PERSIST_DIR = "./data/faiss_index"
    METADATA_DIR = "./data/metadata"
    DOCUMENTS_DIR = "./documents"
    
    # Query Configuration
    TOP_K = 10
    SIMILARITY_THRESHOLD = 0.5
    
    # LLM Configuration
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 4000
    LLM_CONTEXT_SIZE = 8192
    
    # System Prompt
    SYSTEM_PROMPT = """You are an intelligent assistant for engineering document analysis.
You have access to engineering reports, their content, and rich metadata.

CAPABILITIES:
- Answer questions about document content (challenges, solutions, timelines, technical details)
- Query metadata (projects, dates, document types, teams, status)
- Analyze trends across multiple documents
- Extract lessons learned and best practices
- Track project timelines and milestones

INSTRUCTIONS:
1. Use retrieved document content to answer questions about specific details
2. Leverage metadata when answering questions about projects, dates, or document organization
3. Provide comprehensive answers with citations
4. Reference specific documents, sections, dates, or project names
5. If asked about metadata (projects, dates, types), include that information
6. Cross-reference information across multiple documents when relevant
7. Extract and summarize key insights, challenges, and solutions

FORMAT YOUR RESPONSES:
- Start with a direct answer
- Provide supporting details from documents
- Cite sources (document names)
- Include relevant metadata (dates, projects, etc.)
- Use bullet points for clarity when listing multiple items

Remember: You have both the document content AND metadata. Use them together for comprehensive answers.
"""


# ==================== METADATA MANAGER ====================
class MetadataManager:
    """Manages document metadata"""
    
    def __init__(self, metadata_dir: str):
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        self.metadata_file = os.path.join(metadata_dir, "documents_metadata.json")
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load metadata from file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_metadata_from_filename(self, filename: str) -> Dict:
        """Extract metadata from filename patterns"""
        metadata = {
            'filename': filename,
            'file_type': Path(filename).suffix.lower(),
            'indexed_date': datetime.now().isoformat(),
        }
        
        # Extract project name (e.g., "ProjectName_Report.pdf" -> "ProjectName")
        if '_' in filename:
            parts = filename.split('_')
            metadata['project_name'] = parts[0].strip()
        
        # Extract date patterns (YYYY-MM-DD, Q1, Q2, etc.)
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        quarter_pattern = r'Q[1-4]'
        
        date_match = re.search(date_pattern, filename)
        if date_match:
            metadata['document_date'] = date_match.group()
        
        quarter_match = re.search(quarter_pattern, filename)
        if quarter_match:
            metadata['quarter'] = quarter_match.group()
        
        # Document type keywords
        doc_types = {
            'report': ['report', 'summary'],
            'lesson': ['lesson', 'learnt', 'learned'],
            'upgrade': ['upgrade', 'migration'],
            'status': ['status', 'update'],
            'plan': ['plan', 'strategy'],
        }
        
        filename_lower = filename.lower()
        for doc_type, keywords in doc_types.items():
            if any(kw in filename_lower for kw in keywords):
                metadata['document_type'] = doc_type
                break
        
        return metadata
    
    def add_document_metadata(self, filename: str, custom_metadata: Optional[Dict] = None):
        """Add or update document metadata"""
        auto_metadata = self.extract_metadata_from_filename(filename)
        
        if custom_metadata:
            auto_metadata.update(custom_metadata)
        
        self.metadata[filename] = auto_metadata
        self.save_metadata()
    
    def get_document_metadata(self, filename: str) -> Dict:
        """Get metadata for a specific document"""
        return self.metadata.get(filename, {})
    
    def search_by_metadata(self, **filters) -> List[str]:
        """Search documents by metadata filters"""
        matching_docs = []
        
        for filename, meta in self.metadata.items():
            match = True
            for key, value in filters.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
            
            if match:
                matching_docs.append(filename)
        
        return matching_docs
    
    def get_all_projects(self) -> List[str]:
        """Get list of all projects"""
        projects = set()
        for meta in self.metadata.values():
            if 'project_name' in meta:
                projects.add(meta['project_name'])
        return sorted(list(projects))
    
    def get_all_document_types(self) -> List[str]:
        """Get list of all document types"""
        doc_types = set()
        for meta in self.metadata.values():
            if 'document_type' in meta:
                doc_types.add(meta['document_type'])
        return sorted(list(doc_types))
    
    def get_statistics(self) -> Dict:
        """Get metadata statistics"""
        return {
            'total_documents': len(self.metadata),
            'projects': self.get_all_projects(),
            'document_types': self.get_all_document_types(),
            'file_types': list(set(m.get('file_type', '') for m in self.metadata.values())),
        }


# ==================== INTELLIGENT DOCUMENT SYSTEM ====================
class IntelligentDocumentSystem:
    """Main intelligent document analysis system"""
    
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.metadata_manager = MetadataManager(Config.METADATA_DIR)
        self.is_initialized = False
        self.num_documents = 0
    
    def initialize(self) -> Tuple[bool, str]:
        """Initialize the system"""
        try:
            # Setup LLM
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=Config.AWS_REGION
            )
            
            self.llm = Bedrock(
                model=Config.CLAUDE_MODEL_ID,
                client=bedrock_client,
                streaming=False,
                model_kwargs={
                    "temperature": Config.LLM_TEMPERATURE,
                    "max_tokens": Config.LLM_MAX_TOKENS
                },
                context_size=Config.LLM_CONTEXT_SIZE,
                extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
            )
            
            # Setup embeddings
            self.embed_model = HuggingFaceEmbedding(model_name=Config.EMBEDDING_MODEL)
            
            # Configure LlamaIndex
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            
            self.is_initialized = True
            return True, "✅ System initialized successfully!"
            
        except Exception as e:
            return False, f"❌ Initialization error: {str(e)}"
    
    def load_documents(self) -> Tuple[bool, str, List[Document]]:
        """Load documents and extract metadata"""
        try:
            if not os.path.exists(Config.DOCUMENTS_DIR):
                os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
                return False, f"Created {Config.DOCUMENTS_DIR}. Please add documents.", []
            
            reader = SimpleDirectoryReader(
                input_dir=Config.DOCUMENTS_DIR,
                recursive=True,
                required_exts=[".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".csv", ".md"],
                filename_as_id=True
            )
            
            documents = reader.load_data(show_progress=False)
            
            if not documents:
                return False, "No documents found.", []
            
            # Enhance documents with rich metadata
            for doc in documents:
                filename = doc.metadata.get("file_name", "unknown")
                
                # Extract and add metadata
                self.metadata_manager.add_document_metadata(filename)
                doc_metadata = self.metadata_manager.get_document_metadata(filename)
                
                # Merge metadata into document
                doc.metadata.update(doc_metadata)
                
                # Add content-based metadata
                doc.metadata["content_length"] = len(doc.text)
                doc.metadata["has_content"] = len(doc.text.strip()) > 0
            
            self.num_documents = len(documents)
            return True, f"✅ Loaded {len(documents)} documents with metadata!", documents
            
        except Exception as e:
            return False, f"❌ Error loading documents: {str(e)}", []
    
    def build_index(self, documents: List[Document]) -> Tuple[bool, str]:
        """Build vector index with metadata"""
        try:
            os.makedirs(Config.PERSIST_DIR, exist_ok=True)
            
            # Build index
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=False
            )
            
            # Persist
            self.index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
            
            num_nodes = len(self.index.docstore.docs)
            
            return True, f"✅ Index built! {num_nodes} chunks from {self.num_documents} documents."
            
        except Exception as e:
            return False, f"❌ Error building index: {str(e)}"
    
    def load_existing_index(self) -> Tuple[bool, str]:
        """Load existing index"""
        try:
            docstore_path = os.path.join(Config.PERSIST_DIR, "docstore.json")
            if not os.path.exists(docstore_path):
                return False, "No existing index found."
            
            storage_context = StorageContext.from_defaults(persist_dir=Config.PERSIST_DIR)
            self.index = load_index_from_storage(storage_context)
            
            num_nodes = len(self.index.docstore.docs)
            
            return True, f"✅ Loaded index with {num_nodes} chunks."
            
        except Exception as e:
            return False, f"❌ Error loading index: {str(e)}"
    
    def create_engines(self) -> Tuple[bool, str]:
        """Create query and chat engines"""
        try:
            if not self.index:
                return False, "Index not loaded."
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=Config.TOP_K,
            )
            
            # Create query engine with metadata support
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=Config.TOP_K,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=Config.SIMILARITY_THRESHOLD)
                ],
                response_mode=ResponseMode.COMPACT,
            )
            
            # Create chat engine
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=retriever,
                memory=memory,
                verbose=False,
                system_prompt=Config.SYSTEM_PROMPT,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=Config.SIMILARITY_THRESHOLD)
                ],
            )
            
            return True, "✅ Engines created successfully!"
            
        except Exception as e:
            return False, f"❌ Error creating engines: {str(e)}"
    
    def query_with_metadata_context(self, question: str, metadata_filter: Optional[Dict] = None) -> Dict:
        """Query with optional metadata filtering"""
        try:
            # If metadata filter provided, add context
            enhanced_question = question
            
            if metadata_filter:
                filter_context = []
                if 'project_name' in metadata_filter:
                    filter_context.append(f"Focus on project: {metadata_filter['project_name']}")
                if 'document_type' in metadata_filter:
                    filter_context.append(f"Document type: {metadata_filter['document_type']}")
                
                if filter_context:
                    enhanced_question = f"{question}\n\nContext: {', '.join(filter_context)}"
            
            response = self.query_engine.query(enhanced_question)
            
            # Extract sources with metadata
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    filename = node.metadata.get('file_name', 'Unknown')
                    doc_metadata = self.metadata_manager.get_document_metadata(filename)
                    
                    sources.append({
                        'filename': filename,
                        'score': float(node.score) if hasattr(node, 'score') else 0.0,
                        'excerpt': node.text[:400] + "..." if len(node.text) > 400 else node.text,
                        'metadata': doc_metadata,
                        'full_text': node.text
                    })
            
            return {
                'answer': str(response),
                'sources': sources,
                'error': None
            }
            
        except Exception as e:
            return {
                'answer': None,
                'sources': [],
                'error': f"Query error: {str(e)}"
            }
    
    def chat(self, message: str) -> Dict:
        """Chat with the system"""
        try:
            response = self.chat_engine.chat(message)
            
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    filename = node.metadata.get('file_name', 'Unknown')
                    doc_metadata = self.metadata_manager.get_document_metadata(filename)
                    
                    sources.append({
                        'filename': filename,
                        'score': float(node.score) if hasattr(node, 'score') else 0.0,
                        'excerpt': node.text[:400] + "..." if len(node.text) > 400 else node.text,
                        'metadata': doc_metadata,
                        'full_text': node.text
                    })
            
            return {
                'answer': str(response.response),
                'sources': sources,
                'error': None
            }
            
        except Exception as e:
            return {
                'answer': None,
                'sources': [],
                'error': f"Chat error: {str(e)}"
            }
    
    def analyze_metadata(self, query_type: str) -> Dict:
        """Analyze metadata for insights"""
        stats = self.metadata_manager.get_statistics()
        
        if query_type == "projects":
            return {
                'type': 'projects',
                'data': stats['projects'],
                'count': len(stats['projects'])
            }
        elif query_type == "document_types":
            return {
                'type': 'document_types',
                'data': stats['document_types'],
                'count': len(stats['document_types'])
            }
        elif query_type == "overview":
            return {
                'type': 'overview',
                'data': stats,
                'total_documents': stats['total_documents']
            }
        
        return {'type': 'unknown', 'data': {}}
    
    def reset_chat(self):
        """Reset chat memory"""
        if self.chat_engine:
            self.chat_engine.reset()
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'total_chunks': 0,
            'storage_location': Config.PERSIST_DIR,
            'vector_store': 'FAISS'
        }
        
        if self.index:
            stats['total_chunks'] = len(self.index.docstore.docs)
        
        # Add metadata stats
        meta_stats = self.metadata_manager.get_statistics()
        stats.update({
            'total_documents': meta_stats['total_documents'],
            'projects': meta_stats['projects'],
            'document_types': meta_stats['document_types'],
        })
        
        return stats


# ==================== STREAMLIT UI ====================

st.set_page_config(
    page_title="Intelligent Document System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f8ff; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #45a049; }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
    }
    .metadata-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.85em;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1 { color: #2196F3; }
    h2, h3 { color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = IntelligentDocumentSystem()
    st.session_state.initialized = False
    st.session_state.index_built = False
    st.session_state.engines_created = False
    st.session_state.chat_history = []
    st.session_state.metadata_filter = {}

# Sidebar
with st.sidebar:
    st.title("🧠 Intelligent System")
    
    # Setup
    st.subheader("1️⃣ Initialize")
    if st.button("🚀 Initialize System", use_container_width=True):
        with st.spinner("Initializing..."):
            success, message = st.session_state.system.initialize()
            if success:
                st.session_state.initialized = True
                st.success(message)
            else:
                st.error(message)
    
    if st.session_state.initialized:
        st.success("✅ System Ready")
    
    st.divider()
    
    # Index Management
    st.subheader("2️⃣ Document Index")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Load", use_container_width=True, disabled=not st.session_state.initialized):
            with st.spinner("Loading..."):
                success, message = st.session_state.system.load_existing_index()
                if success:
                    st.session_state.index_built = True
                    st.success(message)
                else:
                    st.warning(message)
    
    with col2:
        if st.button("🔨 Build", use_container_width=True, disabled=not st.session_state.initialized):
            with st.spinner("Building..."):
                success, message, documents = st.session_state.system.load_documents()
                if success:
                    with st.spinner("Indexing..."):
                        success, message = st.session_state.system.build_index(documents)
                        if success:
                            st.session_state.index_built = True
                            st.success(message)
    
    st.divider()
    
    # Create Engines
    st.subheader("3️⃣ Create Engines")
    if st.button("⚡ Create", use_container_width=True, disabled=not st.session_state.index_built):
        with st.spinner("Creating engines..."):
            success, message = st.session_state.system.create_engines()
            if success:
                st.session_state.engines_created = True
                st.success(message)
    
    if st.session_state.engines_created:
        st.success("✅ Ready!")
    
    st.divider()
    
    # Metadata Filters
    if st.session_state.engines_created:
        st.subheader("🔍 Filters")
        
        stats = st.session_state.system.get_stats()
        
        # Project filter
        if stats.get('projects'):
            selected_project = st.selectbox(
                "Filter by Project",
                ["All Projects"] + stats['projects']
            )
            if selected_project != "All Projects":
                st.session_state.metadata_filter['project_name'] = selected_project
            elif 'project_name' in st.session_state.metadata_filter:
                del st.session_state.metadata_filter['project_name']
        
        # Document type filter
        if stats.get('document_types'):
            selected_type = st.selectbox(
                "Filter by Type",
                ["All Types"] + stats['document_types']
            )
            if selected_type != "All Types":
                st.session_state.metadata_filter['document_type'] = selected_type
            elif 'document_type' in st.session_state.metadata_filter:
                del st.session_state.metadata_filter['document_type']
        
        if st.session_state.metadata_filter:
            st.markdown(f"""
            <div class="metadata-box">
            <b>Active Filters:</b><br>
            {', '.join(f'{k}: {v}' for k, v in st.session_state.metadata_filter.items())}
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Stats
    if st.session_state.index_built:
        st.subheader("📊 Statistics")
        stats = st.session_state.system.get_stats()
        
        st.metric("Documents", stats.get('total_documents', 0))
        st.metric("Chunks", stats.get('total_chunks', 0))
        
        if stats.get('projects'):
            with st.expander("📂 Projects"):
                for project in stats['projects']:
                    st.write(f"• {project}")
        
        if stats.get('document_types'):
            with st.expander("📄 Document Types"):
                for doc_type in stats['document_types']:
                    st.write(f"• {doc_type}")

# Main content
st.title("🧠 Intelligent Document Analysis System")
st.markdown("Query engineering reports with advanced content and metadata search")

if not st.session_state.engines_created:
    st.info("""
    **🚀 Welcome to the Intelligent Document Analysis System!**
    
    This system provides:
    - ✅ Full-text search across all documents
    - ✅ Metadata filtering (projects, dates, types)
    - ✅ Cross-document analysis
    - ✅ Lesson learned extraction
    - ✅ Timeline and milestone tracking
    
    **Get Started:**
    1. Initialize the system
    2. Build/Load the index
    3. Create engines
    4. Start asking questions!
    
    **Supported formats:** PDF, DOCX, PPTX, XLSX, TXT, CSV, MD
    """)
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Content Questions:**
        - What were the main challenges in the Siebel project?
        - List all lessons learned from platform upgrades
        - What database migration issues were encountered?
        
        **Metadata Questions:**
        - What projects are in the reports?
        - Show me all Q3 documents
        - What types of reports do we have?
        
        **Cross-Document Analysis:**
        - Compare challenges across all upgrade projects
        - What common solutions were used?
        - Summarize all project timelines
        """)

# Chat interface
if st.session_state.engines_created:
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><b>👤 You:</b><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><b>🧠 Assistant:</b><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
            
            # Show sources with metadata
            if message.get('sources'):
                with st.expander(f"📚 {len(message['sources'])} Sources"):
                    for i, source in enumerate(message['sources'], 1):
                        metadata = source.get('metadata', {})
                        st.markdown(f"""
                        <div class="source-box">
                        <b>{i}. {source['filename']}</b> (Score: {source['score']:.3f})<br>
                        <div class="metadata-box">
                        Project: {metadata.get('project_name', 'N/A')} | 
                        Type: {metadata.get('document_type', 'N/A')} | 
                        Date: {metadata.get('document_date', 'N/A')}
                        </div>
                        <small>{source['excerpt']}</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    st.divider()
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input("💬 Ask anything:", key="user_input", 
                                   placeholder="Ask about documents, metadata, or cross-reference information...",
                                   label_visibility="collapsed")
    
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        with st.spinner("🤔 Analyzing..."):
            response = st.session_state.system.chat(user_input)
            
            if response['error']:
                st.error(response['error'])
            else:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response['answer'],
                    'sources': response['sources']
                })
        
        st.rerun()
    
    # Quick actions
    if len(st.session_state.chat_history) == 0:
        st.markdown("### 💡 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 List Documents"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'List all available documents with their projects and types'
                })
                st.rerun()
        
        with col2:
            if st.button("📊 Show Projects"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'What projects are covered in these documents?'
                })
                st.rerun()
        
        with col3:
            if st.button("🎯 Find Lessons"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'Extract all lessons learned from the documents'
                })
                st.rerun()
        
        with col4:
            if st.button("⚠️ Show Challenges"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': 'What were the main challenges across all projects?'
                })
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    🧠 <b>Intelligent Document Analysis System</b><br>
    Content + Metadata • Cross-Document Analysis • Powered by Claude & LlamaIndex
</div>
""", unsafe_allow_html=True)
