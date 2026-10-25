"""
make_kaggle_upload.py
---------------------
Packs what Kaggle needs for the offline guideline search and the R1 counter
into kaggle_upload/offline-rag-kr2/ and kaggle_upload/offline-rag-kr2.zip.
Upload the zip as a private Kaggle dataset.

Included:
  - guideline_index.py, hybrid_bdc_counter.py, README.md
  - guideline_chunks.jsonl and guideline_embeddings.npy (the built index)
  - models/all-MiniLM-L6-v2/ (PyTorch copy of the embedding model, 90 MB;
    downloaded from Hugging Face if it is missing)
  - sanity_queries.json: SANITY_QUERIES with the top-3 chunk IDs this machine
    returns, so day1_kaggle_check.ipynb can confirm Kaggle gives the same
Left out:
  - guidelines/ (the PDFs are only needed to rebuild; the Sphere PDF may not
    be posted online without Sphere's permission)
  - local_chroma_db/ (Kaggle has no chromadb; it uses the .npy vectors)
  - pdf_search_tool.py (needs crewai, which Kaggle does not have)

Usage:
    python guideline_index.py        # build the index first
    python make_kaggle_upload.py
"""

import json
import os
import shutil
import urllib.request

import guideline_index as gi

script_dir = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(script_dir, "kaggle_upload", "offline-rag-kr2")

MODEL_REPO = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "vocab.txt", "special_tokens_map.json"]

FILES = ["guideline_index.py", "hybrid_bdc_counter.py", "README.md",
         "guideline_chunks.jsonl", "guideline_embeddings.npy"]

SANITY_QUERIES = [
    "what does destroyed mean in the xBD damage labels",
    "minor damage roof elements missing visible cracks",
    "building completely collapsed, scorched or no longer present",
    "how to classify building damage from satellite imagery",
    "possibly damaged class when image quality is poor",
    "how to tell the damage grade of a masonry house",
    "reinforced concrete frame building with collapsed ground floor",
    "most buildings destroyed in the area, very destructive earthquake",
    "search and rescue priority for collapsed buildings",
    "assessment search and rescue levels ASR",
    "worksite triage of collapsed structures",
    "how much covered living space per person in a shelter",
    "emergency shelter for displaced families after a disaster",
    "household items to distribute after disaster",
    "reconstruction and long-term recovery of housing",
    "7 destroyed, 12 damaged buildings; search and rescue priority; shelter",
]


def ensure_model() -> None:
    os.makedirs(gi.TORCH_MODEL_DIR, exist_ok=True)
    for name in MODEL_FILES:
        path = os.path.join(gi.TORCH_MODEL_DIR, name)
        if not os.path.isfile(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{MODEL_REPO}/{name}", path)


def main():
    for path in (gi.CHUNKS_PATH, gi.EMBEDDINGS_PATH):
        if not os.path.isfile(path):
            raise SystemExit(f"Missing {path}. Run `python guideline_index.py` first.")
    ensure_model()

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    for name in FILES:
        shutil.copy2(os.path.join(script_dir, name), OUT_DIR)
    shutil.copytree(gi.TORCH_MODEL_DIR, os.path.join(OUT_DIR, "models", "all-MiniLM-L6-v2"))

    expected = [{"query": q, "top3": [r["id"] for r in gi.search_guidelines(q, k=3)]}
                for q in SANITY_QUERIES]
    with open(os.path.join(OUT_DIR, "sanity_queries.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)

    zip_path = shutil.make_archive(OUT_DIR, "zip", root_dir=OUT_DIR)
    print(f"Search backend used for sanity_queries.json: {gi.search_backend()}")
    print(f"Folder: {OUT_DIR}")
    print(f"Zip:    {zip_path} ({os.path.getsize(zip_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
