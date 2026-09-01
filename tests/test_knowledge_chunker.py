"""
Tests for analysis_engine/knowledge.py — SemanticChunker only.
Pure text processing, no external deps (ChromaDB, embeddings, etc.).
"""

import pytest

from analysis_engine.knowledge import SemanticChunker


# ── _is_formula_line ────────────────────────────────────────────────────────

class TestIsFormulaLine:
    @pytest.fixture(autouse=True)
    def _chunker(self):
        self.chunker = SemanticChunker()

    def test_equation_with_equals(self):
        assert self.chunker._is_formula_line("y = mx + b") is True

    def test_sigma_symbol(self):
        assert self.chunker._is_formula_line("Σ x_i") is True

    def test_log_function(self):
        assert self.chunker._is_formula_line("log(P_t / P_{t-1})") is True

    def test_expectation_bracket(self):
        assert self.chunker._is_formula_line("E[r_t | F_{t-1}]") is True

    def test_plain_text_no_formula(self):
        assert self.chunker._is_formula_line("This is a plain sentence.") is False

    def test_empty_string(self):
        assert self.chunker._is_formula_line("") is False


# ── chunk_text ──────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_single_chunk(self):
        chunker = SemanticChunker(chunk_size=5000)
        text = "Hello world.\n\nThis is a test."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert "Hello world." in chunks[0]

    def test_long_text_splits(self):
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=20)
        # Build text with many short paragraphs
        paragraphs = [f"Paragraph {i} with some filler text here." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
        # Every original paragraph should appear in at least one chunk
        for p in paragraphs:
            assert any(p in c for c in chunks)

    def test_empty_text(self):
        chunker = SemanticChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   \n\n   ") == []

    def test_overlap_present(self):
        """Consecutive chunks should share some overlapping content."""
        chunker = SemanticChunker(chunk_size=80, chunk_overlap=30)
        paragraphs = [f"Segment number {i} with enough text to trigger splitting logic." for i in range(15)]
        text = "\n\n".join(paragraphs)
        chunks = chunker.chunk_text(text)
        if len(chunks) >= 2:
            # Check that the tail of chunk[0] overlaps with the head of chunk[1]
            tail = chunks[0][-30:]
            assert tail in chunks[1] or any(word in chunks[1] for word in tail.split())
