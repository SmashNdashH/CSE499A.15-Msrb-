"""
pdf_search_tool.py
------------------
Copy of support/Ridita_Integration_of_VLM__/DisasTeller/pdf_search_tool.py,
made offline. The original is unchanged.

Changes from the original:
  - Gemini embeddings (API key + internet) -> local all-MiniLM-L6-v2 on CPU,
    via the index that guideline_index.py builds.
  - EMS-98 only, cut every 500 characters -> every PDF in guidelines/, split
    by section.
  - Similarity search only -> hybrid search (embeddings + BM25 keywords), so
    exact terms such as "possible damage" or "ASR level" are found.
  - Each result starts with a citation ID such as [EMS98-intensity-X], so a
    report can cite the chunks it used.

Build the index once before using the tool:
    python guideline_index.py
"""

from crewai.tools import tool

from guideline_index import format_results, search_guidelines


@tool
def offline_pdf_search_tool(query: str) -> str:
    """Offline semantic search over local disaster guidelines (EMS-98 and others).
    Each result starts with a citation ID in square brackets; cite these IDs."""
    return format_results(search_guidelines(query, k=3))
