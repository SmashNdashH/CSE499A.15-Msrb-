"""
guideline_index.py
------------------
Offline guideline index for DisasTeller (RAG v1). No API keys, no internet
after the first run.

Build: each PDF listed in GUIDELINES is split at its headings (lines set in
bold or in a larger font than the body text). Long sections are split again
at paragraph breaks to about TARGET_WORDS words. Every chunk gets a citation
ID such as EMS98-1.9, EMS98-masonry-grade-4 or EMS98-intensity-X. Chunks are written
to guideline_chunks.jsonl and embedded into a local Chroma folder with
Chroma's built-in all-MiniLM-L6-v2 (ONNX, runs on CPU; the 80 MB model is
downloaded once to ~/.cache/chroma).

Search: search_guidelines(query, k) returns the k best chunks with their IDs,
merging the embedding ranking with a BM25 keyword ranking.

The build also saves the chunk vectors to guideline_embeddings.npy. Where
chromadb is not installed (Kaggle), search uses those vectors with numpy and
embeds the query with a PyTorch copy of the same model from
models/all-MiniLM-L6-v2 (transformers + torch only), so it needs no pip
install and no internet. Set GUIDELINE_BACKEND=numpy to force that path.

Usage:
    # Print detected sections without building the index
    python guideline_index.py --dry_run

    # Build (or rebuild) the index
    python guideline_index.py

    # Query the built index
    python guideline_index.py --query "collapse of masonry buildings"
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
GUIDELINES_DIR = os.path.join(script_dir, "guidelines")
CHROMA_DB_DIR = os.path.join(script_dir, "local_chroma_db")
CHUNKS_PATH = os.path.join(script_dir, "guideline_chunks.jsonl")
EMBEDDINGS_PATH = os.path.join(script_dir, "guideline_embeddings.npy")
TORCH_MODEL_DIR = os.path.join(script_dir, "models", "all-MiniLM-L6-v2")
COLLECTION_NAME = "guidelines"

# One entry per PDF in guidelines/. Pages are 1-based and inclusive; use them
# to skip covers, contents pages and reference lists. Optional keys:
#   "headings": "numbered"  only numbered headings start a section
#   "min_size": 9.5         drop text smaller than this (captions, tables)
#   "grade_groups": {...}   name damage grades by the table they are in
# A .txt entry (text a PDF only holds as an image) becomes one chunk named
# by its "chunk_id" and headed by its "section"; lines starting with # are skipped.
GUIDELINES = [
    {
        "file": "EMS98_Original_english__earthquake.pdf",
        "prefix": "EMS98",
        "title": "EMS-98",
        "first_page": 8,
        "last_page": None,
        # Damage grades are listed once for masonry and once for RC buildings
        "grade_groups": {"masonry": "masonry", "reinforced concrete": "rc"},
    },
    {
        # arxiv.org/pdf/1911.09296. Section 4 is the Joint Damage Scale our labels use.
        "file": "xBD_Gupta2019_1911.09296.pdf",
        "prefix": "XBD",
        "title": "xBD paper (Gupta et al., 2019)",
        "first_page": 1,
        "last_page": 8,  # p. 9 is references
        "headings": "numbered",
    },
    {
        # The scale itself (Figure 4, p. 4) is an image in the PDF, so it is typed out here
        "file": "xBD_Joint_Damage_Scale_Figure4.txt",
        "prefix": "XBD",
        "title": "xBD paper (Gupta et al., 2019)",
        "chunk_id": "joint-damage-scale",
        "section": "xBD Joint Damage Scale: the damage labels 0-3 used in xBD and xView2",
        "first_page": 4,
        "last_page": 4,
    },
    {
        # un-spider.org; the damage-grading guideline Copernicus EMS Rapid Mapping uses
        "file": "IWG_SEM_Building_Damage_Assessment_v1.0.pdf",
        "prefix": "IWGSEM",
        "title": "IWG-SEM Building Damage Assessment v1.0 (Copernicus EMS)",
        "first_page": 3,  # pp. 1-2 are the cover and contents
        "last_page": None,
        "headings": "numbered",
        "min_size": 9.5,  # the 8-9 pt text is a jumbled class table and figure captions
    },
    {
        # insarag.org/guidelines, 2020 edition
        "file": "INSARAG_2020_Vol_III_Operational_Field_Guide.pdf",
        "prefix": "INSARAG3",
        "title": "INSARAG Guidelines 2020, Vol. III Operational Field Guide",
        "first_page": 7,  # pp. 1-6 are the cover, contents and abbreviations
        "last_page": None,
        "headings": "numbered",
    },
    {
        "file": "INSARAG_2020_Vol_II_Manual_B_Operations.pdf",
        "prefix": "INSARAG2B",
        "title": "INSARAG Guidelines 2020, Vol. II Manual B Operations",
        "first_page": 6,  # pp. 1-5 are the cover, contents and abbreviations
        "last_page": 55,  # pp. 56-58 are a change log and a list of online annexes
        "headings": "numbered",
    },
    {
        # Fourth edition. The official spherestandards.org link refuses scripted
        # downloads; this copy came from globalprotectioncluster.org (458 pages,
        # metadata author "Sphere"). Only the Shelter and Settlement chapter is used.
        "file": "Sphere_Handbook_2018_EN.pdf",
        "prefix": "SPHERE",
        "title": "Sphere Handbook 2018, Shelter and Settlement",
        "first_page": 275,  # pp. 272-274 are the chapter cover and contents
        "last_page": 324,   # pp. 325-329 are references and further reading
        "headings": "numbered",  # sections = the 7 standards; "Key actions" etc. stay in the text
    },
]

TARGET_WORDS = 250     # split long sections near this size
MAX_WORDS = 320        # never let a chunk grow past this
MIN_SECTION_WORDS = 25  # shorter sections are folded into the next one
MARGIN = 0.08          # top/bottom share of a page that holds running headers and footers
RRF_K = 60             # reciprocal rank fusion constant (the usual value)

SECTION_NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?(?:\s+|$)")
GRADE = re.compile(r"^Grade\s+(\d)\b", re.IGNORECASE)
ROMAN = re.compile(r"^([IVX]{1,4})\.\s")
APPENDIX = re.compile(r"^Appendix\s+(\d+)\b")


# ─── PDF → lines ────────────────────────────────────────────────────────────

def _read_lines(pdf_path: str, first_page: int, last_page, min_size: float = 0):
    """Return (page, block, text, bold, size) for every non-empty text line.

    Running headers and footers are dropped: lines in the top or bottom
    MARGIN of the page that are no larger than body text and repeat on 2+
    pages (digits ignored), or that are far smaller than body text. Large
    titles near the top ("Appendix 1", "Appendix 2") are kept. So is text
    smaller than min_size, used to skip figure captions and tables.
    """
    import fitz  # PyMuPDF; only needed to build the index

    doc = fitz.open(pdf_path)
    last = last_page or doc.page_count
    lines = []
    for pno in range(first_page - 1, last):
        page_height = doc[pno].rect.height
        for bno, block in enumerate(doc[pno].get_text("dict")["blocks"]):
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join("".join(s["text"] for s in line["spans"]).split())
                bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans)
                size = max(s["size"] for s in spans)
                y = line["bbox"][1] / page_height
                in_margin = y < MARGIN or y > 1 - MARGIN
                if size >= min_size:
                    lines.append((pno + 1, bno, text, bold, size, in_margin))

    def norm(text):
        return re.sub(r"\d+", "#", text.lower())

    body_size = _body_size(lines)
    margin_pages = defaultdict(set)
    for page, _, text, _, size, in_margin in lines:
        if in_margin and size <= body_size + 0.5:
            margin_pages[norm(text)].add(page)

    def running(line):
        _, _, text, _, size, in_margin = line
        return in_margin and (len(margin_pages[norm(text)]) >= 2 or size <= body_size - 2.5)

    return [line[:5] for line in lines if not running(line)]


def _body_size(lines) -> float:
    sizes = Counter()
    for _, _, text, _, size, *_ in lines:
        sizes[round(size, 1)] += len(text)
    return sizes.most_common(1)[0][0]


def _looks_like_heading(text: str, bold: bool, size: float, body_size: float) -> bool:
    if len(text) > 100:
        return False
    if not re.search(r"[A-Za-z]", text) and not SECTION_NUM.match(text):
        return False
    return bold or size >= body_size + 1.5


# ─── lines → sections → chunks ──────────────────────────────────────────────

def _split_sections(lines, numbered_only: bool = False):
    """Group lines into sections: {"heading", "parent", "page_start", "page_end", "paragraphs"}.

    With numbered_only, only numbered headings ("3.2 Operations") start a
    section; other bold lines ("Key actions") stay in the section text.
    """
    body_size = _body_size(lines)
    sections = []
    current = {"heading": "", "page_start": lines[0][0], "page_end": lines[0][0], "paragraphs": []}
    current["parent"] = ""
    parent = ""  # last heading that is not a damage grade
    heading_lines = 0
    heading_key = None
    last_key = None

    for page, block, text, bold, size in lines:
        if text.isdigit() and not bold:
            continue  # page number
        key = (page, block)
        heading = _looks_like_heading(text, bold, size, body_size)
        numbered = any(p.match(text) for p in (SECTION_NUM, GRADE, ROMAN, APPENDIX))
        # A heading wraps onto at most two more lines of its own text block
        continues_heading = heading and 0 < heading_lines < 3 and key == heading_key and not numbered
        starts_heading = heading and not continues_heading and not text.endswith(".") \
            and (not numbered_only or SECTION_NUM.match(text) or APPENDIX.match(text))

        if continues_heading:
            current["heading"] += " " + text
            heading_lines += 1
        elif starts_heading:
            sections.append(current)
            if current["heading"] and not GRADE.match(current["heading"]):
                parent = current["heading"]
            current = {"heading": text, "parent": parent, "page_start": page, "page_end": page,
                       "paragraphs": []}
            heading_lines = 1
            heading_key = key
        else:
            heading_lines = 0
            if key != last_key or not current["paragraphs"]:
                current["paragraphs"].append(text)
            else:
                current["paragraphs"][-1] += " " + text
            last_key = key
            current["page_end"] = page
    sections.append(current)

    for s in sections:
        # Re-join words hyphenated across line breaks: "struc- tural" -> "structural"
        s["paragraphs"] = [re.sub(r"(\w)- (\w)", r"\1\2", p) for p in s["paragraphs"]]
    return [s for s in sections if s["heading"] or s["paragraphs"]]


def _fold_short_sections(sections):
    """Prepend sections with almost no body text to the section that follows.

    Damage grades and intensity degrees are short but stand alone. A bare
    table title right before a damage grade is dropped, because build_chunks
    puts it in front of every grade heading under it.
    """
    out, carry = [], None
    for s in sections:
        if carry:
            bare_title = not carry["paragraphs"] and GRADE.match(s["heading"]) \
                and s["parent"] == carry["heading"]
            if not bare_title:
                s["paragraphs"] = [carry["heading"]] + carry["paragraphs"] + s["paragraphs"]
                s["page_start"] = carry["page_start"]
            carry = None
        n_words = sum(len(p.split()) for p in s["paragraphs"])
        stands_alone = GRADE.match(s["heading"]) or ROMAN.match(s["heading"])
        if n_words < MIN_SECTION_WORDS and not stands_alone:
            carry = s
        else:
            out.append(s)
    if carry and out:
        out[-1]["paragraphs"] += [carry["heading"]] + carry["paragraphs"]
    return out


def _section_id(heading: str, parent: str = "", grade_groups=None) -> str:
    m = GRADE.match(heading)
    if m:
        group = next((g for word, g in (grade_groups or {}).items() if word in parent.lower()), "")
        return f"{group}-grade-{m.group(1)}" if group else f"grade-{m.group(1)}"
    for pattern, fmt in ((SECTION_NUM, "{}"), (ROMAN, "intensity-{}"), (APPENDIX, "appendix-{}")):
        m = pattern.match(heading)
        if m:
            return fmt.format(m.group(1))
    words = re.findall(r"[A-Za-z0-9]+", heading.lower())[:4]
    return "-".join(words) or "intro"  # text before the first heading


def _split_paragraphs(paragraphs):
    """Pack paragraphs into parts of about TARGET_WORDS words, splitting overlong ones at sentences."""
    pieces = []
    for p in paragraphs:
        if len(p.split()) <= MAX_WORDS:
            pieces.append(p)
        else:
            pieces.extend(re.split(r"(?<=[.;:])\s+", p))

    parts, current = [], []
    for piece in pieces:
        n_current = sum(len(x.split()) for x in current)
        if current and n_current + len(piece.split()) > MAX_WORDS or n_current >= TARGET_WORDS:
            parts.append(current)
            current = []
        current.append(piece)
    if current:
        parts.append(current)
    return ["\n".join(part) for part in parts]


def _text_chunk(guideline: dict) -> dict:
    with open(os.path.join(GUIDELINES_DIR, guideline["file"]), encoding="utf-8") as f:
        body = "\n".join(line.strip() for line in f if line.strip() and not line.startswith("#"))
    text = f"{guideline['section']}\n{body}"
    return {
        "id": f"{guideline['prefix']}-{guideline['chunk_id']}",
        "doc": guideline["title"],
        "section": guideline["section"],
        "page_start": guideline["first_page"],
        "page_end": guideline["last_page"],
        "n_words": len(text.split()),
        "text": text,
    }


def build_chunks(guideline: dict) -> list:
    if guideline["file"].endswith(".txt"):
        return [_text_chunk(guideline)]
    pdf_path = os.path.join(GUIDELINES_DIR, guideline["file"])
    lines = _read_lines(pdf_path, guideline["first_page"], guideline["last_page"],
                        guideline.get("min_size", 0))
    sections = _split_sections(lines, guideline.get("headings") == "numbered")
    # Numbered headings by number, so "3.2 Operations" can show its chapter "3 Management"
    by_number = {}
    for s in sections:
        m = SECTION_NUM.match(s["heading"])
        if m:
            by_number.setdefault(m.group(1), s["heading"])
    sections = _fold_short_sections(sections)

    chunks, seen = [], Counter()
    for s in sections:
        section_id = _section_id(s["heading"], s["parent"], guideline.get("grade_groups"))
        base = f"{guideline['prefix']}-{section_id}"
        seen[base] += 1
        if seen[base] > 1:
            base = f"{base}-{seen[base]}"
        heading = s["heading"]
        if GRADE.match(heading) and s["parent"]:
            # "Grade 5: Destruction" alone does not say which building type it is for
            heading = f"{s['parent']}: {heading}"
        m = SECTION_NUM.match(heading)
        if m and "." in m.group(1):
            chapter = by_number.get(m.group(1).rsplit(".", 1)[0])
            if chapter:
                heading = f"{chapter} > {heading}"
        parts = _split_paragraphs(s["paragraphs"])
        for i, body in enumerate(parts, start=1):
            chunk_id = base if len(parts) == 1 else f"{base}-p{i}"
            text = f"{heading}\n{body}" if heading else body
            chunks.append({
                "id": chunk_id,
                "doc": guideline["title"],
                "section": heading,
                "page_start": s["page_start"],
                "page_end": s["page_end"],
                "n_words": len(text.split()),
                "text": text,
            })
    return chunks


# ─── Index build and search ─────────────────────────────────────────────────

def _chroma_client():
    import chromadb
    from chromadb.config import Settings

    # Telemetry off: Chroma otherwise sends anonymous usage events over the network
    return chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=Settings(anonymized_telemetry=False))


def build_index(chunks: list) -> None:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    embedding_function = DefaultEmbeddingFunction()
    vectors = np.asarray(embedding_function([c["text"] for c in chunks]), dtype=np.float32)
    np.save(EMBEDDINGS_PATH, vectors)  # same order as guideline_chunks.jsonl

    client = _chroma_client()
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c[k] for k in ("doc", "section", "page_start", "page_end")} for c in chunks],
        embeddings=vectors,
    )


class _TorchMiniLM:
    """all-MiniLM-L6-v2 run with transformers + torch. It pools and normalises the
    way Chroma's ONNX copy does (256-token cut, mean over tokens, L2 norm), so its
    query vectors match the stored chunk vectors."""

    def __init__(self, model_dir: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"No embedding model at {model_dir}.")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval()

    def __call__(self, texts: list) -> np.ndarray:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with self.torch.no_grad():
            tokens = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.torch.nn.functional.normalize(pooled, dim=1).numpy()


STOPWORDS = set("a an and are as at be by does for from how in is it of on or that the this "
                "to was what when which who with".split())


def _tokenize(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class _BM25:
    """Keyword ranking, so exact terms ("possible damage", "ASR") are not lost
    when the embedding of a long chunk averages them away."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(_tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(texts)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list:
        terms = _tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            out.append(sum(self.idf[t] * doc[t] * (self.k1 + 1) / (doc[t] + norm)
                           for t in terms if t in doc))
        return out


_index = None


def _chroma_available() -> bool:
    if os.environ.get("GUIDELINE_BACKEND") == "numpy" or not os.path.isdir(CHROMA_DB_DIR):
        return False
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def _load_index():
    """Return (backend name, semantic ranking function, chunks, BM25)."""
    global _index
    if _index is None:
        if not os.path.isfile(CHUNKS_PATH):
            raise FileNotFoundError(f"No index at {CHUNKS_PATH}. Run `python guideline_index.py` first.")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

        if _chroma_available():
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            collection = _chroma_client().get_collection(
                COLLECTION_NAME, embedding_function=DefaultEmbeddingFunction())
            backend = "chroma"

            def semantic(query, n):
                return collection.query(query_texts=[query], n_results=n)["ids"][0]
        else:
            vectors = np.load(EMBEDDINGS_PATH)
            if len(vectors) != len(chunks):
                raise ValueError(f"{EMBEDDINGS_PATH} has {len(vectors)} vectors for {len(chunks)} chunks. "
                                 "Rebuild with `python guideline_index.py`.")
            embed = _TorchMiniLM(TORCH_MODEL_DIR)
            backend = "numpy + torch"

            def semantic(query, n):
                similarity = vectors @ embed([query])[0]
                return [chunks[i]["id"] for i in np.argsort(-similarity)[:n]]

        _index = (backend, semantic, chunks, _BM25([c["text"] for c in chunks]))
    return _index


def search_backend() -> str:
    """Which search path is in use: "chroma" (laptop) or "numpy + torch" (Kaggle)."""
    return _load_index()[0]


def search_guidelines(query: str, k: int = 3, pool: int = 50) -> list:
    """Return the k best chunks for `query` as dicts with id, doc, section, pages, text, score.

    Hybrid search: the embedding ranking and the BM25 keyword ranking (top
    `pool` of each) are merged by reciprocal rank fusion.
    """
    _, semantic, chunks, bm25 = _load_index()
    semantic_ids = semantic(query, pool)
    keyword_scores = bm25.scores(query)
    keyword_ids = [chunks[i]["id"] for i in sorted(range(len(chunks)), key=lambda i: -keyword_scores[i])
                   if keyword_scores[i] > 0][:pool]

    fused = Counter()
    for ranking in (semantic_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1 / (RRF_K + rank)

    by_id = {c["id"]: c for c in chunks}
    return [
        {**{key: by_id[cid][key] for key in ("id", "doc", "section", "page_start", "page_end", "text")},
         "score": round(score, 4)}
        for cid, score in fused.most_common(k)
    ]


def format_results(results: list) -> str:
    """Format search results so each one starts with its citation ID in square brackets."""
    blocks = []
    for r in results:
        pages = f"p. {r['page_start']}" if r["page_start"] == r["page_end"] else \
            f"pp. {r['page_start']}-{r['page_end']}"
        blocks.append(f"[{r['id']}] ({r['doc']}, {pages})\n{r['text']}")
    return "\n\n".join(blocks)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build or query the offline guideline index.")
    parser.add_argument("--dry_run", action="store_true", help="Print sections, do not build.")
    parser.add_argument("--query", type=str, help="Query the built index instead of building.")
    parser.add_argument("--k", type=int, default=3, help="Results per query (default: 3).")
    args = parser.parse_args()

    if args.query:
        print(format_results(search_guidelines(args.query, k=args.k)))
        return

    chunks = []
    for g in GUIDELINES:
        doc_chunks = build_chunks(g)
        words = sorted(c["n_words"] for c in doc_chunks)
        print(f"{g['file']}: {len(doc_chunks)} chunks, "
              f"median {words[len(words) // 2]} words, max {words[-1]}")
        chunks += doc_chunks

    if args.dry_run:
        for c in chunks:
            print(f"  {c['id']:<28} p{c['page_start']:>3}  {c['n_words']:>4}w  {c['section'][:70]}")
        return

    build_index(chunks)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_DB_DIR}")
    print(f"Chunk list written to {CHUNKS_PATH}")
    print(f"Chunk vectors written to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    main()
