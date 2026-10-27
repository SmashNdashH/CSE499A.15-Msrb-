# Plan 2: Agentic Vision RAG (patch-level inspection)

**Author:** Tamanna (Member B) · **Group:** CSE499B Group 5 · **Folder:** `Essential_task_delegation/Tamanna_task_2/Plan_2/`
**Based on:** ImageRAG (IEEE GRSM 2025) and Change-Agent (IEEE TGRS 2024)

## Status (27 Oct 2026)

**Demo, not an evaluation.** Plan 2 has no measured results yet. Abrar's review found that the notebooks' saved outputs didn't match a recorded run of the committed code, and that several numbers in this folder were never measured. They have been removed:

- the results table that was in this README (building-count MAE, class agreement, EMS-98 consistency and VRAM for four settings), and the viva answers that quoted it;
- the typed-in and fallback numbers in both notebooks' summary and chart cells;
- both charts and `google_paligemma_benchmark.csv` in `outputs/`.

What was wrong and what has been fixed is in the correction at the top of [Audit_Response_and_Scientific_Fixes.md](Audit_Response_and_Scientific_Fixes.md). Next comes Part B, one small test against ground truth, once Abrar has checked this part.

## The idea

In CSE499A, fine-tuning made our 7B worse at counting buildings than the base model (the reference points are in [Task_Ledger.md §1](../../Task_Ledger.md)). Plan 2 tries one idea for grading buildings: the U-Net finds each building, and the VLM looks at a full-resolution crop of one building at a time instead of the whole tile. Whether this helps has not been measured.

## How the agent works

1. **Segmenter:** in the notebooks, R9, the D4 champion recipe (U-Net + ResNet-50), finds the buildings and gives each one a class: Intact, Damaged or Destroyed. Images are normalised with ImageNet's mean and std, as in training, and run at full size.
2. **Crops:** up to 3 buildings per tile, Destroyed and Damaged first, each a 256 × 256 window of the original image.
3. **VLM:** two questions per building, each built for that building. First its damage class on its own, in one word; then a description, an EMS-98 grade and a rescue priority. The VLM is never told the segmenter's class.
4. **Report:** per building, the segmenter's class, the VLM's class and grade, and whether the two classes agree. Agreement between two models is not accuracy. A ground-truth mask, when one is given, is used only to score the classes, and never replaces the segmenter's prediction ([agent.py](agent.py)).

The EMS-98 table in `config.py` is not put into the prompt: the VLM grades from its own knowledge.

## Kaggle runs

Each notebook was run once with Restart & Run All on Kaggle, on the `abrarmohammedtanzim/disasterm3-mirror` dataset, and then committed with Save & Run All, which runs it again in a fresh kernel. The files here are the committed versions and their outputs. Both runs used the same 10 tiles, drawn from the team's validation split (colour photos only), the same R9 checkpoint, the same prompts and the same library versions; only the VLM differs.

| Notebook | VLM | Kaggle version | Output files |
|---|---|---|---|
| [Agentic-vision-rag.ipynb](Agentic-vision-rag.ipynb) | our fine-tuned 7B, `AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP` | committed 27 Oct 2026; link to add | `outputs/finetuned_demo_*` |
| [Base_model_demo.ipynb](Base_model_demo.ipynb) | the base model, `Qwen/Qwen2.5-VL-7B-Instruct` | committed 27 Oct 2026; link to add | `outputs/base_demo_*` |

**Demo, not an evaluation.** The outputs compare the VLM's class with the segmenter's class: agreement between two models, not accuracy. None of their numbers is a result of Plan 2.

Each run wrote:
- `*_demo_tiles.csv`: per tile, the buildings found, the buildings inspected, agreements, conflicts, unparsed answers and seconds;
- `*_demo_crops.csv`: per inspected building, both classes, the VLM's two answers, the grade and the priority;
- `*_demo_summary.csv`, `*_demo_config.json`: the run's totals, settings, prompts and library versions;
- `*_demo_dashboard.png`, `*_demo_chart.png`: the dashboard for the first tile, and a chart of this run's measured values only;
- `*_demo_huggingface_repos.json`: Kaggle's record of the model revisions the run downloaded.

The base notebook was `BaseModel_and_PaliGemma2_Benchmark.ipynb` until 27 Oct 2026, when PaliGemma was dropped (the edge model is L2's job). The fine-tuned notebook was committed before the rename, so its introduction still names its companion by the old file name.

The older files in `outputs/` (`base_model_benchmark.csv`, `batch_validation_benchmark.csv`, `master_tri_vlm_comparison.csv`, `inspection_dashboard_sample.png`, both charts and `google_paligemma_benchmark.csv`) were removed: no recorded run of the committed code produced them.

### Running a notebook on Kaggle
1. **Settings:** GPU T4 ×2 (as in both runs), internet on.
2. **Input:** `abrarmohammedtanzim/disasterm3-mirror`.
3. **R9's checkpoint**, which is private. Either a Kaggle secret `HF_TOKEN` that can read `AbrarAlam/disasterm3-d4-R9` (Add-ons > Secrets, switched on for the notebook), or the output of Abrar's `d4-champion-model-R9` notebook added as an input (it holds `d4_R9/final_model.pth`). Without either, the notebook stops at the segmenter cell and says which is missing; it doesn't fall back to another model.
4. Restart & Run All, then Save Version, then Save & Run All (Commit).

## Files

| File | What it is |
|---|---|
| `Agentic-vision-rag.ipynb`, `Base_model_demo.ipynb` | The two demo notebooks, self-contained for Kaggle |
| `agent.py` | The agent, with the same logic as the notebooks' agent cell |
| `tools.py` | The segmentation, crop and EMS-98 tools used by the scripts |
| `vlm_engine.py` | Loads the fine-tuned or the base Qwen2.5-VL-7B in 4-bit |
| `config.py` | Paths, the model registry, the classes and the EMS-98 table |
| `visualizer.py` | The annotated map and the dashboard for the scripts |
| `run_pipeline.py` | A command-line run on one image, or `--self_test` on a synthetic tile |
| `evaluate_benchmark.py` | A command-line run over a folder of tiles; never run |
| `requirements.txt` | The scripts' dependencies |
| `Plan_2_Visual_RAG_Patch_Zooming.md` | The plan (added 17 Oct 2026, last changed 20 Oct). Its numbers are planning estimates and figures from papers, not measurements |
| `imageRAG_RP.md` | The ImageRAG paper |
| `Audit_Response_and_Scientific_Fixes.md` | The response to the first review, with a correction at the top |
| `outputs/` | The output files of the Kaggle runs |

## The command-line scripts

The scripts haven't been run on real data. `run_pipeline.py --self_test` runs the agent on a synthetic tile whose mask stands in for the segmenter's output. It still downloads the VLM; offline, the VLM runs in mock mode and no VLM number is reported.

Before relying on the scripts:
- `tools.py` loads checkpoints with `strict=False`, and builds an untrained U-Net when it finds no checkpoint. The notebooks use their own strict loader instead.
- `evaluate_benchmark.py` picks tiles by file name, not from the team's validation split, and can't tell colour photos from radar tiles.
