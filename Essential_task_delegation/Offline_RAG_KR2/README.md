# Offline RAG + KR2 counting (Ridita)

This folder has my work for the next update:

1. An offline search over disaster guidelines, for DisasTeller (RAG v1).
2. Building counting with the R1 model (KR2).

Rule: files from other folders are copied here.
The originals are never changed.


## Files

**guideline_index.py**
- Reads the guideline PDFs and cuts them into sections.
- Saves the sections in a local search index (Chroma).
- Searches the index. Uses meaning (embeddings) and keywords (BM25) together.
- Runs on CPU. Needs no API key and no internet.

**pdf_search_tool.py**
- Offline copy of DisasTeller's `pdf_search_tool.py`.
- A CrewAI tool that the agents can call.
- Every result starts with a citation ID, for example `[EMS98-masonry-grade-5]`.

**hybrid_bdc_counter.py**
- Copy of Abrar's counter
  (`support/Abrar/Implementation_DisasterM3/hybrid_bdc_counter.py`).
- New option `--mode r1` for R1's labels:
  0 = Background, 1 = Intact, 2 = Damaged, 3 = Destroyed.
- The CSV also gets one count per class (intact, damaged, destroyed).

**guidelines/**
- The source documents. See the list below.

**guideline_chunks.jsonl**, **guideline_embeddings.npy** and **local_chroma_db/**
- Made by `guideline_index.py`.
- Build them again after you change anything in `guidelines/`.

**models/all-MiniLM-L6-v2/**
- A PyTorch copy of the search model (90 MB), used on Kaggle.
- Downloaded from Hugging Face by `make_kaggle_upload.py` if it is missing.

**make_kaggle_upload.py**
- Packs everything Kaggle needs into `kaggle_upload/offline-rag-kr2.zip`.

**day1_kaggle_check.ipynb**
- Kaggle notebook that proves the search and the counter work there
  with the internet off.

`models/` and `kaggle_upload/` are in this folder's `.gitignore`
(too big for git, and both can be made again).


## Source documents

**EMS98**: European Macroseismic Scale 1998
- Pages used: 8 to the end.
- Copied from `support/Ridita_Integration_of_VLM__/DisasTeller/Guidelines/`.

**XBD**: xBD paper (Gupta et al., 2019)
- Pages used: 1 to 8 (page 9 is references).
- From https://arxiv.org/pdf/1911.09296

**XBD-joint-damage-scale**: the damage labels 0 to 3 used by xBD
- This table is a picture in the PDF (Figure 4, page 4), so the program
  cannot read it.
- It is typed out word for word in
  `guidelines/xBD_Joint_Damage_Scale_Figure4.txt`.

**IWGSEM**: IWG-SEM Building Damage Assessment v1.0
- The damage guide that Copernicus EMS Rapid Mapping uses.
- Pages used: 3 to 8.
- From https://www.un-spider.org/sites/default/files/IWG_SEM_Guidelines_Building%20Damage%20Assessment_v1.0.pdf

**INSARAG3**: INSARAG Guidelines 2020, Volume III (Operational Field Guide)
- Pages used: 7 to the end.
- From https://www.insarag.org/guidelines

**INSARAG2B**: INSARAG Guidelines 2020, Volume II, Manual B (Operations)
- Pages used: 6 to 55.
- From https://www.insarag.org/guidelines

**SPHERE**: Sphere Handbook 2018, Shelter and Settlement chapter
- Pages used: 275 to 324.
- The official site (spherestandards.org) blocks downloads by script,
  so this copy is from globalprotectioncluster.org.
- Important: you may use it for study and research, but you need
  Sphere's permission to put it online.
  So do NOT push `Sphere_Handbook_2018_EN.pdf` to a public GitHub repo.


## How to run

List the sections the program finds (does not build anything):

    python guideline_index.py --dry_run

Build the index (takes about a minute):

    python guideline_index.py

Search the index:

    python guideline_index.py --query "collapsed ground floor"

Count buildings in R1 masks:

    python hybrid_bdc_counter.py --mask_dir masks/ --mode r1 --output counts.csv


## Running on Kaggle with internet turned off

Kaggle does not have `chromadb`. So on Kaggle the search uses the saved
vectors (`guideline_embeddings.npy`) and a PyTorch copy of the same model.
It needs only `numpy`, `torch` and `transformers`, which Kaggle already has.
Nothing to install, nothing to download.

1. On the laptop, make the upload file:

       python guideline_index.py        # only if you changed guidelines/
       python make_kaggle_upload.py

2. On Kaggle: **Datasets → New Dataset**, upload
   `kaggle_upload/offline-rag-kr2.zip`, name it `offline-rag-kr2`,
   keep it **Private**.
3. **Code → New Notebook → File → Import Notebook**, choose
   `day1_kaggle_check.ipynb`.
4. Right panel: Accelerator **None**, Internet **Off**,
   **+ Add Input** → your `offline-rag-kr2` dataset.
5. **Run All**. You should see `Internet is OFF`,
   `Search backend: numpy + torch`, `16/16` and `Counter OK`.
6. **Save Version** to keep the output as proof.

In later notebooks (Day 3), add the same dataset and use:

    import glob, os, sys
    sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/guideline_index.py", recursive=True)[0]))
    import guideline_index as gi
    print(gi.format_results(gi.search_guidelines("collapsed buildings, rescue priority", k=3)))
