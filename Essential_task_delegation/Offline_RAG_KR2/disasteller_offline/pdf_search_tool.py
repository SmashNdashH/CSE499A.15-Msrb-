"""
pdf_search_tool.py
------------------
Copy of support/Ridita_Integration_of_VLM__/DisasTeller/pdf_search_tool.py, made
offline. The original is unchanged.

The original embedded EMS-98 with Gemini (API key + internet) into gemini_chroma_db/.
This copy keeps the tool's name, so DisasterManagement_teamwork_simulation.py still
imports offline_pdf_search_tool, but the tool now searches the offline guideline index
in the folder above (Offline_RAG_KR2/): guideline_index.py and the files it builds.

It imports guideline_index from that folder instead of copying the index here:
  - guideline_index.py finds its built files (chunks, vectors, Chroma folder, model)
    next to itself, so DisasTeller searches exactly the index the retrieval test checks;
  - after `python guideline_index.py`, the one rebuild serves both, and no second copy
    can go out of date;
  - the backend choice stays the same (Chroma on the laptop, numpy + torch on Kaggle).
Set GUIDELINE_INDEX_DIR to use a copy of that folder elsewhere, e.g. a Kaggle input.
"""

import os
import sys

from crewai.tools import tool

GUIDELINE_INDEX_DIR = os.environ.get("GUIDELINE_INDEX_DIR") or \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isfile(os.path.join(GUIDELINE_INDEX_DIR, "guideline_index.py")):
    raise FileNotFoundError(f"No guideline_index.py in {GUIDELINE_INDEX_DIR}. "
                            "Set GUIDELINE_INDEX_DIR to the Offline_RAG_KR2 folder.")
sys.path.append(GUIDELINE_INDEX_DIR)

from guideline_index import format_results, search_guidelines  # noqa: E402


@tool
def offline_pdf_search_tool(query: str) -> str:
    """Offline search over disaster guidelines: EMS-98 and IWG-SEM (building damage
    grades), the xBD damage scale, INSARAG (search and rescue teams, ASR levels,
    worksite triage) and Sphere (shelter, living space, household items, reconstruction).
    Each result starts with a citation ID in square brackets; cite these IDs."""
    return format_results(search_guidelines(query, k=3))
