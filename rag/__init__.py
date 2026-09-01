"""
RAG module for the AFML Quant Pipeline.
Re-exports from analysis_engine.knowledge (canonical implementation).
"""
try:
    from analysis_engine.knowledge import (
        KnowledgeBase,
        create_knowledge_base,
        SemanticChunker,
        GeminiEmbeddings,
    )

    __all__ = [
        "KnowledgeBase",
        "create_knowledge_base",
        "SemanticChunker",
        "GeminiEmbeddings",
    ]
except ImportError:
    # Graceful fallback if analysis_engine is not available
    import warnings
    warnings.warn(
        "analysis_engine.knowledge not available — RAG functionality disabled",
        ImportWarning,
    )
    __all__ = []
