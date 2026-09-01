"""
AFML Quant Pipeline - RAG Knowledge Base
ChromaDB + LangChain / Vertex embeddings over quantitative-finance references.

Optional layer. Every third-party package imported below ships in
requirements-extras.txt, not in requirements.txt, so each import is guarded:
this module imports cleanly without them and the classes that need a missing
package raise on construction with an explicit message. ``SemanticChunker`` is
pure Python and always available.

Nothing here mutates the environment at import time. If the system cache
directory is not writable, call ``use_local_cache_dirs()`` explicitly before
constructing a ``KnowledgeBase``.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

try:
    import chromadb
    from chromadb.config import Settings
    import chromadb.utils.embedding_functions as embedding_functions
    HAS_CHROMADB = True
except ImportError:  # pragma: no cover - exercised only without the extras
    chromadb = None
    Settings = None
    embedding_functions = None
    HAS_CHROMADB = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    HAS_GOOGLE_GENAI = True
except ImportError:  # pragma: no cover - exercised only without the extras
    GoogleGenerativeAIEmbeddings = None
    HAS_GOOGLE_GENAI = False

try:
    import vertexai
    import google.auth
    from vertexai.language_models import TextEmbeddingModel
    HAS_VERTEX_SDK = True
except ImportError:
    HAS_VERTEX_SDK = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_HF = True
except ImportError:
    HAS_HF = False

# Document processing (PDF / EPUB ingestion only).
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:  # pragma: no cover - exercised only without the extras
    PdfReader = None
    HAS_PDF = False

try:
    from ebooklib import epub
    from bs4 import BeautifulSoup
    HAS_EPUB = True
except ImportError:  # pragma: no cover - exercised only without the extras
    epub = None
    BeautifulSoup = None
    HAS_EPUB = False

# Project defaults
GCP_PROJECT = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

load_dotenv()


def use_local_cache_dirs() -> None:
    """Point the Chroma/ONNX model cache at ``./.cache`` under the CWD.

    Opt-in, and deliberately never called at import time: it rewrites
    ``HOME`` and ``XDG_CACHE_HOME`` for the whole process. Call it by hand
    before constructing a ``KnowledgeBase`` if the system cache directory is
    locked or read-only.
    """
    os.environ['XDG_CACHE_HOME'] = str(Path.cwd() / ".cache")
    os.environ['HOME'] = str(Path.cwd())


def _require(flag: bool, package: str, feature: str) -> None:
    """Raise a pointed ImportError when an optional extra is missing."""
    if not flag:
        raise ImportError(
            f"{feature} needs {package}, an optional extra. "
            f"Install it with: pip install -r requirements-extras.txt"
        )


class SemanticChunker:
    """
    Semantic chunking that preserves mathematical formulas and context.
    Uses sentence boundaries and formula detection.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _is_formula_line(self, line: str) -> bool:
        """Detect if a line contains mathematical notation."""
        formula_indicators = ['=', '∑', '∫', '∏', 'σ', 'μ', 'Σ', 'log', 'exp', 
                              'E[', 'Var[', 'Cov[', '^', '_', '\\frac', '\\sum']
        return any(ind in line for ind in formula_indicators)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks, preserving formulas."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size, save current and start new
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + "\n\n" + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class VertexSDKEmbeddings:
    """Wrapper for Vertex AI SDK Embeddings."""
    def __init__(self, model_name: str = "text-embedding-004", project=None, location=None):
        _require(HAS_VERTEX_SDK, "google-cloud-aiplatform", "Vertex AI embeddings")
        # Force quota project on credentials
        credentials, _ = google.auth.default()
        if hasattr(credentials, 'with_quota_project'):
            credentials = credentials.with_quota_project(project)
        
        vertexai.init(project=project, location=location, credentials=credentials)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Vertex AI has a limit of 250 inputs per request usually, but let's assume batching is handled or list is small
        # Actually, let's process in chunks of 5 to be safe and avoid rate limits
        embeddings = []
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results = self.model.get_embeddings(batch)
            embeddings.extend([r.values for r in results])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.get_embeddings([text])[0].values

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

class GeminiEmbeddings:
    """Gemini embedding model wrapper for ChromaDB using LangChain or Vertex SDK."""
    
    def __init__(self, model_name: str = "text-embedding-004"):
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key and HAS_GOOGLE_GENAI:
            self.model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        elif HAS_VERTEX_SDK and GCP_PROJECT:
             # Try Vertex
             try:
                print(f"Attempting Vertex AI SDK embeddings ({GCP_PROJECT})")
                self.model = VertexSDKEmbeddings(
                    model_name=model_name,
                    project=GCP_PROJECT,
                    location=GCP_LOCATION
                )
             except (Exception, BaseException) as e:
                 print(f"Vertex Init Failed (Broad Catch): {e}")
                 print("Falling back to ChromaDB default embeddings (ONNX)")
                 self.model = self._default_embedding_function()
        else:
            print("Using ChromaDB default embeddings (ONNX): no credentials found")
            self.model = self._default_embedding_function()
    
    @staticmethod
    def _default_embedding_function():
        _require(HAS_CHROMADB, "chromadb", "The default (ONNX) embedding function")
        return embedding_functions.DefaultEmbeddingFunction()
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if hasattr(self.model, 'embed_documents'):
             return self.model.embed_documents(input)
        # DefaultEmbeddingFunction (chroma) is callable directly
        return self.model(input)


class KnowledgeBase:
    """
    RAG Knowledge Base for quantitative finance books.
    Uses ChromaDB with Gemini embeddings via LangChain.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        _require(HAS_CHROMADB, "chromadb", "KnowledgeBase")
        self.persist_directory = persist_directory
        self.chunker = SemanticChunker()
        self.embedding_fn = GeminiEmbeddings()
        
        # Initialize ChromaDB
        # Prevent pydantic from reading broken .env file
        # Hack: Pydantic crashes if it finds a locked .env file in CWD.
        # We temporarily switch CWD to / to avoid this during Settings init.
        cwd = os.getcwd()
        try:
             os.chdir("/")
             try:
                 settings = Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False
                )
                 # Hack to avoid .env reading if pydantic supports it
                 if hasattr(settings, '_env_file'):
                     settings._env_file = None
             except Exception:
                 # Fallback
                 settings = Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False
                )
        finally:
            os.chdir(cwd)

        self.client = chromadb.PersistentClient(path=persist_directory, settings=settings)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="quant_finance_books",
            metadata={"description": "López de Prado and quant finance reference"}
        )
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        _require(HAS_PDF, "PyPDF2", "PDF ingestion")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    
    def _extract_text_from_epub(self, epub_path: str) -> str:
        """Extract text from EPUB file."""
        _require(HAS_EPUB, "ebooklib and beautifulsoup4", "EPUB ingestion")
        book = epub.read_epub(epub_path)
        text = ""
        
        for item in book.get_items():
            if item.get_type() == epub.EpubHtml:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n\n"
        
        return text
    
    def ingest_document(self, file_path: str, source_name: Optional[str] = None) -> int:
        """
        Ingest a PDF or EPUB document into the knowledge base.
        Returns number of chunks created.
        """
        path = Path(file_path)
        source = source_name or path.stem
        
        # Extract text based on file type
        if path.suffix.lower() == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        elif path.suffix.lower() == '.epub':
            text = self._extract_text_from_epub(file_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text)
        
        # Generate embeddings and store
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_chunk_{i}"
            embedding = self.embedding_fn([chunk])[0]
            
            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": source, "chunk_index": i}]
            )
        
        print(f"Ingested {len(chunks)} chunks from {source}")
        return len(chunks)
    
    def ingest_directory(self, directory: str) -> int:
        """Ingest all PDF and EPUB files from a directory."""
        total_chunks = 0
        path = Path(directory)
        
        for file_path in path.glob("*"):
            if file_path.suffix.lower() in ['.pdf', '.epub']:
                total_chunks += self.ingest_document(str(file_path))
        
        return total_chunks
    
    def query(self, question: str, n_results: int = 5) -> List[dict]:
        """
        Query the knowledge base.
        Returns relevant chunks with metadata.
        """
        # Generate query embedding
        query_embedding = self.embedding_fn([question])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted
    
    def get_context_for_prompt(self, question: str, n_results: int = 3) -> str:
        """Get formatted context string for LLM prompts."""
        results = self.query(question, n_results)
        
        context_parts = []
        for r in results:
            source = r['metadata'].get('source', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{r['content']}")
        
        return "\n\n---\n\n".join(context_parts)


# Convenience function for quick setup
def create_knowledge_base(books_directory: str = None) -> KnowledgeBase:
    """
    Create and optionally populate a knowledge base.
    
    Usage:
        kb = create_knowledge_base("/path/to/books")
        results = kb.query("What is the Triple Barrier Method?")
    """
    kb = KnowledgeBase()
    
    if books_directory:
        kb.ingest_directory(books_directory)
    
    return kb


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        books_dir = sys.argv[1]
        kb = create_knowledge_base(books_dir)
        
        # Test query
        results = kb.query("Explain fractional differentiation for stationarity")
        for r in results:
            print(f"\n[{r['metadata']['source']}]")
            print(r['content'][:500] + "...")
    else:
        print("Usage: python knowledge_base.py <books_directory>")
