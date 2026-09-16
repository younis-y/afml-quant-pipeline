
"""
Explainability script for the AFML Quant Pipeline.
Queries the RAG Knowledge Base to explain financial concepts.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import KnowledgeBase (this triggers the environment/cache setup)
from analysis_engine.knowledge import KnowledgeBase

def explain(query: str, n_results: int = 3):
    """Query the knowledge base and print results."""
    print(f"\nQuerying knowledge base for: '{query}'...")
    print("-" * 60)
    
    # Initialize KB pointing to the root chroma_db
    # We assume 'chroma_db' is in the current working directory or specific path
    kb = KnowledgeBase(persist_directory=str(PROJECT_ROOT / "chroma_db"))
    
    try:
        results = kb.query(query, n_results=n_results)
    except Exception as e:
        print(f"Error querying database: {e}")
        return

    if not results:
        print("No results found.")
        return

    for i, res in enumerate(results):
        source = res['metadata'].get('source', 'Unknown')
        content = res['content']
        distance = res['distance']
        
        print(f"\nSource {i+1}: {source} (distance: {distance:.4f})")
        print("..." + content.replace("\n", " ")[:300] + "...")
        print("-" * 40)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Explain financial concepts using the Knowledge Base")
    parser.add_argument('--query', type=str, required=True, help="Concept to explain")
    parser.add_argument('--n', type=int, default=3, help="Number of results")
    
    args = parser.parse_args()
    explain(args.query, args.n)

if __name__ == "__main__":
    main()
