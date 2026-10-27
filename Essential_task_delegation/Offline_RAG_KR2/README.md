# Offline RAG + KR2 counting (Ridita)

This folder has my work for the next update:

1. An offline search over disaster guidelines, for DisasTeller (RAG v1).
2. Building counting with the R1 model (KR2).
3. The guideline lookup for T2 (DisasTeller on local models): a copy of
   DisasTeller that is offline by default, and a retrieval test of the search.

Rule: files from other folders are copied here.
The originals are never changed.


## Abrar's review of 26 Oct: what each step did

1. **DisasTeller copy, offline by default** (`disasteller_offline/`): the
   guideline search is linked to this folder's index, and the online switch of
   System_Design_Notes §2 is added. Why: O3 promises a complete report with no
   internet, so offline is the default and the only mode that is graded. See
   "DisasTeller, offline by default".
2. **30 test questions, fixed before any search ran**
   (`retrieval_test/questions.csv`), with the pass rule written down first.
   Why: the reports cite what this search returns, and the search had never
   been measured. Fixing the test first means it can't be fitted to the
   results. See "Retrieval test of the guideline search".
3. **The test run: passed, 27 of 30** (the bar was 24). The meaning ranking
   alone got 20, so the hybrid search stays. Also in "Retrieval test of the
   guideline search".
4. **Kaggle proof of the day-1 check: passed with version 2.** It ran as a
   commit on 27 Oct with Kaggle's Internet setting off: `Internet is OFF`,
   search `16/16` and `Counter OK`, as expected before the run. Version 1 had
   the internet on, so it doesn't count. Why: the graded T2 run is on Kaggle
   with the internet off (System_Design_Notes §2.4), so the search has to work
   there with nothing downloaded. Version 2:
   https://www.kaggle.com/code/riditaafrin/day1-kaggle-check?scriptVersionId=362688031
   (public). See "Running on Kaggle with internet turned off".
5. **Counter:** work on `hybrid_bdc_counter.py` has stopped. Why: H2's table
   already holds R11's buildings one by one (see Files).
6. **Sphere:** the PDF and its sections stay as they are until Abrar decides.
   Why: Sphere's permission is needed to put the handbook online, and the
   repository is public (see Source documents). Checked 27 Oct: the PDF is
   unchanged and kept out of git, and its 61 sections are the same as in the
   first push.
7. **This README:** each step's reasons, the expectation fixed before the test,
   and the result against it.


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
- DisasTeller itself imports `disasteller_offline/pdf_search_tool.py`, which
  searches the same index.

**hybrid_bdc_counter.py**
- Copy of Abrar's counter
  (`support/Abrar/Implementation_DisasterM3/hybrid_bdc_counter.py`).
- New option `--mode r1` for R1's labels:
  0 = Background, 1 = Intact, 2 = Damaged, 3 = Destroyed.
- The CSV also gets one count per class (intact, damaged, destroyed).
- Work on it has stopped (review, 26 Oct): H2's table already holds R11's buildings one by one (117,687 buildings in 2,652 benchmark images); see [H2's README](../Champion_Task/H2-results/H2/README.md).

**guidelines/**
- The source documents. See the list below.

**guideline_chunks.jsonl**, **guideline_embeddings.npy** and **local_chroma_db/**
- Made by `guideline_index.py`.
- Build them again after you change anything in `guidelines/`.
- A search with the Chroma backend rewrites two files in `local_chroma_db/`
  (a lock row in `chroma.sqlite3`, and an index file `length.bin`) without
  changing the data, so git shows them as modified afterwards. Undo it with
  `git checkout -- local_chroma_db/` before committing.

**models/all-MiniLM-L6-v2/**
- A PyTorch copy of the search model (90 MB), used on Kaggle.
- Downloaded from Hugging Face by `make_kaggle_upload.py` if it is missing.

**make_kaggle_upload.py**
- Packs everything Kaggle needs into `kaggle_upload/offline-rag-kr2.zip`.

**day1_kaggle_check.ipynb**
- Kaggle notebook that proves the search and the counter work there
  with the internet off.

**disasteller_offline/**
- Copy of DisasTeller (`support/Ridita_Integration_of_VLM__/DisasTeller/`),
  offline by default. See "DisasTeller, offline by default" below.

**retrieval_test/**
- `questions.csv`: 30 questions a DisasTeller report needs, and the sections
  that answer them. See "Retrieval test of the guideline search" below.
- `run_retrieval_test.py` runs the test and writes `results.csv` (the top 3
  of each question, and whether it is a hit) and `results_summary.json` (the
  counts, the misses and every ranking).

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
- Left as it is until Abrar decides (review, 26 Oct): the PDF and its 61
  SPHERE sections. Those sections hold the chapter's text, and they are in git,
  in `guideline_chunks.jsonl` and in `local_chroma_db/chroma.sqlite3`.


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
4. **Settings** menu: **Turn off internet** (the item then reads "Turn on
   internet"), and **Accelerator → None**. **+ Add Input** → your
   `offline-rag-kr2` dataset.
5. **Run All**. You should see `Internet is OFF`,
   `Search backend: numpy + torch`, `16/16` and `Counter OK`.
6. **Save Version → Save & Run All (Commit)**. Kaggle runs the whole notebook
   again with these settings, and the version keeps its outputs as the proof.
7. When the version has finished, open it and download into `results/` the
   notebook with its outputs, and `day1_kaggle_check.json` from its **Output**
   tab. Never edit an output by hand.
8. **Share** the notebook with the Kaggle account `abrarmohammedtanzim` (or
   make it public), and put the version's link (the address with
   `scriptVersionId=`) under "Kaggle proof" below.

**Kaggle proof (review step 4): version 2, with Internet off, passed.**
https://www.kaggle.com/code/riditaafrin/day1-kaggle-check?scriptVersionId=362688031

Version 2 (27 Oct, committed 09:26, 69 s), downloaded unchanged into
`results/v2_internet_off/` (`day1-kaggle-check.ipynb`, `day1_kaggle_check.json`):
- **Internet was off.** Kaggle's metadata in the notebook says
  `"isInternetEnabled": false`, cell 2 printed `Internet is OFF`, and the JSON
  says `"internet": "off"`. Cell 2's connection attempt took 32 s to fail; in
  version 1 it connected in 0.06 s.
- **The rest passed:** accelerator none (`"accelerator": "none"`,
  `"isGpuEnabled": false`), `chromadb installed: False`,
  `Search backend: numpy + torch`, `16/16` and `Counter OK`. The demo query
  gave the same top 3 as version 1.
- **The same input as version 1:** both used dataset version 21290657. Cell 1
  lists the same files as `kaggle_upload/offline-rag-kr2/`, whose code and index
  files are byte-identical to this folder's (only its README is older).
- Checked: it is a committed run (Kaggle's papermill metadata, 6 code cells run
  in order, no error); every cell's code is the same as
  `day1_kaggle_check.ipynb`; the JSON file is the notebook's printed summary,
  byte for byte.
- The model-loading progress bar in cell 3 is saved at `0/103`: the notebook
  file keeps only the bar's first frame. The next line shows the search ran.
- Link: https://www.kaggle.com/code/riditaafrin/day1-kaggle-check?scriptVersionId=362688031
- The notebook is public: the link opens without a login (checked 27 Oct). The
  `offline-rag-kr2` dataset stays private (Kaggle shows it as
  "[Private Datasource]"), so anyone can see the outputs, but only with access
  to the dataset can they run the notebook again.

Version 1 (27 Oct, committed 08:31, 48 s; its files were checked, then not
kept, because only version 2 is the proof):
- **Internet was on.** The notebook printed `Internet is ON`, the JSON says
  `"internet": "on"`, and Kaggle's own metadata in the notebook says
  `"isInternetEnabled": true`. With the internet on, the run can't show that
  nothing was downloaded, so it isn't the offline proof.
- The rest passed: accelerator none, the `offline-rag-kr2` dataset attached,
  `Search backend: numpy + torch`, `16/16` and `Counter OK`. The demo query's
  top 3 (`EMS98-rc-frame-mexico-city-2`, `EMS98-rc-frame-north-pelopponissos-2`,
  `INSARAG2B-5.8.2-p1`) is also what this PC gives on the upload folder.
- Checked: it is a committed run (Kaggle's papermill metadata, 6 code cells run
  in order); every cell's code is the same as `day1_kaggle_check.ipynb`; the
  JSON file is the notebook's printed summary, byte for byte.

Checked on 27 Oct, before version 1:
- `kaggle_upload/offline-rag-kr2.zip` (made 25 Oct) still matches the index
  and the code byte for byte, and it holds no PDF, so it can be uploaded as it
  is. Only its copy of this README is older, which the notebook doesn't read.
- The notebook's search and counter checks, run on this PC on the unpacked
  upload folder with the Kaggle backend (numpy + torch): `16/16` and
  `Counter OK`.
- The notebook's last cell now asks for Save & Run All (Commit), not Quick Save.

In later notebooks (Day 3), add the same dataset and use:

    import glob, os, sys
    sys.path.insert(0, os.path.dirname(glob.glob("/kaggle/input/**/guideline_index.py", recursive=True)[0]))
    import guideline_index as gi
    print(gi.format_results(gi.search_guidelines("collapsed buildings, rescue priority", k=3)))


## DisasTeller, offline by default (T2 steps 4 and 6)

**What was copied**
- Everything in `support/Ridita_Integration_of_VLM__/DisasTeller/` except
  `gemini_chroma_db/` and `__pycache__/` (no file is over 50 MB): 20 files,
  checked byte for byte against the originals before any edit.
- The six `.txt` reports at the top of `disasteller_offline/` are the Gemini
  run's outputs, copied unchanged, for comparing with later.
- New runs write to `reports/offline/` or `reports/online/`, so they never
  overwrite them. As in the original, the paths are relative to the folder you
  run from, so run it from `disasteller_offline/`.

**How the guideline search is linked**
- `disasteller_offline/pdf_search_tool.py` keeps the tool's name,
  `offline_pdf_search_tool`, so the main script imports it unchanged (line 28).
- It imports `guideline_index.py` from the folder above, instead of copying the
  index into `disasteller_offline/`. Why:
  - `guideline_index.py` finds its built files (chunks, vectors, Chroma folder,
    model) next to itself, so DisasTeller searches exactly the index that the
    retrieval test checks;
  - one rebuild (`python guideline_index.py`) serves both, and no second copy
    can go out of date;
  - the backend choice stays the same: Chroma on the laptop, numpy + torch on
    Kaggle.
- To use a copy of this folder somewhere else (for example a Kaggle input), set
  `GUIDELINE_INDEX_DIR` to it.
- The tool's description now names all six sources, because the agents choose
  their tools by it.

**The switch (System_Design_Notes §2.2)**
- Offline is the default. `DISASTELLER_ONLINE=1` (an environment variable, or a
  line in `.env`) turns online mode on. The script prints which mode it runs in.
- Offline, no agent gets the two web tools, and `requests`, `bs4` and
  `duckduckgo_search` are not even imported.
- Online, the web tools and the original wording of tasks 3–6 come back
  unchanged (checked character for character against the original).
- In both modes:
  - the Emergency service and Assignment teams also get
    `offline_pdf_search_tool` (before, only the Expert team had it);
  - every report starts with `Mode: offline` or `Mode: online`;
  - CrewAI's usage telemetry is off. CrewAI 1.15.2 sends it to
    `telemetry.crewai.com` unless `CREWAI_DISABLE_TELEMETRY` is set.

**The offline wording for tasks 3–6**
- The original sends the agents to the internet in five places: line 140
  (task3), 154 (task4), 176 (task5), 193 (task6) and 202–203 (task6's expected
  output). The review listed 154, 176 and 193. System_Design_Notes §2.1 lists 140
  and 202–203 as well, so all five are switched: an agent told to search, with no
  search tool, calls a tool that isn't there or makes the data up.
- Each task is pointed at the right guideline:
  - task3 (shelters and hospitals): Sphere for shelter (living space and site
    area per person, location and access), and INSARAG for handing rescued
    people over to hospitals;
  - task4 (staff allocation): INSARAG (team types, ASR levels, worksite triage
    and priorities, shifts);
  - task5 (public report): Sphere (shelter, household items) and INSARAG
    (rescue);
  - task6 (reconstruction plan): Sphere (repair, retrofit or rebuild, debris,
    building codes) and EMS-98 (damage grades).
- All four end with the same rule: use only the facts given and the guideline
  search, cite the section IDs, and write "not available offline" for anything
  that needs data the agent does not have: costs, budgets, staff numbers from
  past disasters, population counts or contact details.
- Task6's expected output says "Money needed: not available offline". It asks
  for the order of the steps instead of timelines, so no durations are made up.

**What still calls Gemini (roadmap T2 steps 1–3, not changed yet)**

| File in `disasteller_offline/` | Lines | What | Used by |
|---|---|---|---|
| `DisasterManagement_teamwork_simulation.py` | 36–42; `llm=gemini_llm` at 93, 104, 116, 128 | `LLM(model="gemini/gemini-3.1-flash-lite")`, the LLM of all four agents | every task |
| `local_img_tool.py` | 7, 13–14, 38, 65 | `genai.GenerativeModel("gemini-3.1-flash-lite")` describes the local images | `local_img_interpreter`: Expert and Emergency service teams |
| `global_img_tool.py` | 7, 13–14, 38, 64 | the same, for the global map | `global_img_interpreter`: Alerts, Emergency service and Assignment teams |
| `global_annotation_tool.py` | 5, 15–16, 62, 100 | the same, to place the grade labels on the map | `global_map_annotation`: Expert team |

**Other things to clear before the offline proof run (System_Design_Notes §2.4)**
- Line 37 copies the Gemini key into `GEMINI_API_KEY` and fails if there is
  none, so until T2 step 2 even offline mode needs a Gemini key and the internet.
- The three image tools call `genai.configure` and encode their images as soon
  as they are imported, before any task runs.
- `from langchain.tools import tool as langchain_tool` (line 23) is unused but
  still needs `langchain` installed, and `global_annotation_tool.py` imports
  `openai` without using it.
- On its first run CrewAI asks whether to upload a trace; nothing is sent unless
  the answer is yes.
- On a new machine the Chroma backend downloads its ONNX model once (to
  `~/.cache/chroma`). With the internet off, use `GUIDELINE_BACKEND=numpy` and
  `models/all-MiniLM-L6-v2/`, as on Kaggle.

**Not done here:** the §2.3 rules for online mode (web findings in their own
section, page text treated as data), the H2 facts and T1 hot spots as input
(T2 step 5), and the consistency check (T2 step 7).

**How it was checked.** It can't run end to end yet, because it still needs
Gemini.
- The copy against the original, by sha256, before editing.
- The task wording, by parsing both files (no code was run): online wording
  equals the original for all six tasks, offline wording never mentions the
  internet, tasks 3–6 carry the offline rule, and the output paths are under
  `reports/<mode>/`.
- `py_compile` on both edited files, and an import of the tool, which finds
  `guideline_index` in this folder without loading the index.


## Retrieval test of the guideline search (T2 step 4)

**Why:** DisasTeller's reports cite what this search returns, so it has to find
the section that answers each question a report needs. That had not been
measured: the day-1 check only compares Kaggle's results with the laptop's.

**Expectation, written on 26 Oct before any search was run**
- Pass: at least 24 of the 30 questions have an expected section in the top 3
  of `search_guidelines(q, k=3)`.
- The same 30 questions are also scored with each ranking alone, from the same
  index: the meaning ranking (embeddings, its top 3) and the BM25 keyword
  ranking (top 3 by BM25 score). If the meaning ranking alone gets at least as
  many hits as the hybrid, the simpler search is kept.
- The run uses this PC's default backend (Chroma). The 30 questions are then
  repeated with `GUIDELINE_BACKEND=numpy`, the Kaggle path, and any difference
  in the top 3 is reported.
- The questions and expected IDs are not changed after the results are seen.
  If the test fails, it is reported as failed; any fix is a second run,
  reported separately.

**The questions** (`retrieval_test/questions.csv`: `question`, `expected_ids`,
`source`)
- 30 questions a DisasTeller report needs: damage grades by building type,
  rescue priority and triage, ASR levels, shelter space per person, household
  items, the xBD damage labels and reconstruction.
- Per source: EMS98 5, XBD 4, IWGSEM 4, INSARAG3 4, INSARAG2B 6, SPHERE 7.
- `source` is the guideline the question was written for. `expected_ids` lists,
  separated by `;`, every section whose own text gives the answer, fully or in
  part; in 5 questions one of them comes from another guideline. Sections that
  mention the topic only in passing, or need an inference to give the answer,
  are not listed.
- The IDs were chosen by reading all 405 chunks of `guideline_chunks.jsonl`. A
  plain text search (grep) of that file was used only to check that no other
  chunk also answers. The search under test was not run.
- None of the 30 repeats one of the 16 `SANITY_QUERIES` in
  `make_kaggle_upload.py`, which were run while the index was being built. They
  cover the same topics, though, so the result may be a little optimistic.

**Result (27 Oct): passed, 27 of 30.** From `retrieval_test/results.csv` and
`results_summary.json`, made by `run_retrieval_test.py`. Qn is row n of
`questions.csv`.

| Ranking | Hits | EMS98 | XBD | IWGSEM | INSARAG3 | INSARAG2B | SPHERE |
|---|---|---|---|---|---|---|---|
| Hybrid, `search_guidelines` (graded) | **27/30** | 4/5 | 4/4 | 4/4 | 3/4 | 5/6 | 7/7 |
| Meaning ranking alone | 20/30 | 3/5 | 2/4 | 4/4 | 3/4 | 3/6 | 5/7 |
| BM25 keyword ranking alone | 25/30 | 4/5 | 4/4 | 4/4 | 1/4 | 6/6 | 6/7 |

- **Against the expectation:** the hybrid clears the bar of 24 by 3. The
  meaning ranking alone gets 7 fewer hits, so the hybrid search stays.
- **BM25 alone** gets 25, 2 fewer than the hybrid. The rule fixed beforehand
  only compares the meaning ranking with the hybrid, so this decides nothing.
  The two differ on four questions: the hybrid finds Q21, Q22 and Q28, which
  BM25 misses, and BM25 finds Q16, which the hybrid misses.
- **How sure the gains are** (worked out after the run; it changes no decision):
  an exact sign test on the questions where two rankings disagree (McNemar's
  test). Against the meaning ranking, the hybrid wins 8 such questions and
  loses 1 (p = 0.04). Against BM25 it wins 3 and loses 1 (p = 0.63), so 30
  questions show a clear gain over the meaning ranking, but can't tell the
  hybrid and BM25 apart.
- **Reproduced:** a second run on a copy of the index (27 Oct) gave the same
  `results.csv` byte for byte, and the same summary.
- **The Kaggle path** (`GUIDELINE_BACKEND=numpy`) gives the same top 3 as
  Chroma on all 30 questions, in all three rankings.
- **Where the hits are:** the hybrid puts an expected section first on 16 of
  the 30. A hit needs only one expected section, and three hits return only
  part of the answer: Q14 finds ASR level 4's definition but not level 3's, Q27
  repair and retrofit but not rebuilding, and Q30 a checklist line but not the
  structural check (`SPHERE-2-p6`).

**The three misses**
- **Q1, EMS98** ("A brick house has walls that have broken apart and part of
  its roof has fallen in..."): expected masonry grade 4 or one of its five
  examples. Got `EMS98-unreinforced-masonry-roermond-the` (a grade 2 example),
  `EMS98-masonry-grade-3` and `EMS98-rc-frame-mexico-city` (an RC grade 3
  example), so an agent would likely answer grade 3. The meaning ranking alone
  found it. It matters for task1, which grades buildings with this search.
- **Q16, INSARAG2B** ("One collapsed building has three people confirmed
  alive; at another, neighbours say people are missing..."): expected
  `INSARAG2B-5.8.1-p1`, which says confirmed live victims come first. Got
  `INSARAG2B-5.8.1-p2` (what confirmed and possible live victims mean, without
  the order), `INSARAG2B-6.4` (signals) and `INSARAG2B-5.7.1-p6` (ASR level 4).
  BM25 alone found it.
- **Q23, INSARAG3** ("Must medium and heavy USAR teams keep operations running
  24 hours a day?"): expected `INSARAG3-3.2-p3` ("Medium and Heavy Teams are to
  ensure 24-hour operations"). Got `INSARAG2B-3.8-p1` (a 24-hour team focal
  point), `INSARAG3-2` (the checklists' introduction) and `INSARAG2B-4.3-p1`
  (work cycles and rest). No ranking found it.

No fix was tried. A fix would be a second run, reported separately. These 30
questions have now been seen, so a fix judged only on them would be tuned on
the test; a new question set would be the fair check.
