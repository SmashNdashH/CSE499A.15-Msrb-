# Manual renames inside notebooks that already ran

Written 20 Oct 2026, after your naming decisions:
- R6 is now R1s7. R6 is R1 with seed 7, which would be R1s7 under today's rule.
- "Step" everywhere, never "stage": L2's "stage 1" is now "step 1".
- R16, R16b and R16c keep their names. R16c's steps are renumbered 1–5 (2b → 3, 3a → 4, 3b → 5).
- R16's two parts are called **the finding** (SAM 3 plus the U-Net's missed pieces) and **the grading** (the Siamese grader). The model is **find-then-grade**, not "two-stage".

The files and folders are already renamed, and the notebooks that haven't run yet are already edited. These notebooks have run, and their recorded outputs must stay. Editing a cell's text in VS Code keeps its outputs, as long as you **don't run the cell**, so these edits are yours.

## How to edit, step by step (VS Code)

1. In VS Code's Explorer, open the notebook (double-click it). It opens as cells, not as text.
2. Find the cell: press **Ctrl+F** and type the start of the cell's first line, given below as "starts with".
3. **Markdown cell** (text): double-click it, or click it and press **Enter**, to edit. Change the line. Press **Esc** to stop editing; it shows as text again.
4. **Code cell**: click inside the code and edit the line. Then click outside the cell.
   - **Never press Shift+Enter, Ctrl+Enter or the ▶ button.** Those run the cell and replace its recorded output.
5. **Add a new cell at the top** (R1s7 only): hover your mouse just above the first cell. Click **+ Markdown**, paste the text, then press **Esc**.
6. Save with **Ctrl+S**. VS Code may reformat the file; the cells and outputs stay the same.
7. **Don't change** any name inside quotes that is a column or key, such as `"stage1"`, `"stage1_buildings"` or `"stage1_macro_f1"`. Don't rename code variables either, such as `stage1 = ...` in E3. Saved tables use those names, and other notebooks read them. Each such line below says **leave**.

"Line N" counts lines inside the cell, starting at 1. Only the text shown changes; the rest of the line stays.

---

## 1. `Champion_Task\d4-champion-model-R1s7.ipynb` (was `d4-champion-model-R6.ipynb`)

**Add a new first cell (markdown)**, above the cell that starts with `# DisasterM3 — D4 Champion Model`. Paste:

> **Renamed on 20 Oct 2026: this run was R6.** R6 is R1 with seed 7, which would be R1s7 under today's rule. The outputs below are as it ran, under the name R6 (for example `D4_R6.csv`). Its results are now in `R1s7-results/d4_R1s7/D4_R1s7.csv`.

**Cell starting with `# DisasterM3 — D4 Champion Model + Pre-D4 Ablations`** (markdown):
- line 3: `ablation (R1–R6),` → `ablation (R1–R5 and R1s7, formerly R6),`
- line 30: `| **R6** | seed 7 |` → `| **R1s7** (formerly R6) | seed 7 |`
- line 32: `\|R1 − R6\|` → `\|R1 − R1s7\|`
- line 35: `\|R1 − R6\|` → `\|R1 − R1s7\|`
- line 58: `R2–R6 are independent` → `R2–R5 and R1s7 are independent`
- line 59: `R6 first so the noise bar` → `R1s7 first so the noise bar`

**Cell starting with `# ── Cell 2: Configuration — pick EXPERIMENT`** (code). Change these two lines together, so a re-run still finds its settings:
- line 17: `EXPERIMENT = "R6"             # E0 | R1 | R2 | R3 | R4 | R5 | R6 | R8 | R7` → `EXPERIMENT = "R1s7"           # E0 | R1 | R2 | R3 | R4 | R5 | R1s7 | R8 | R7`
- line 42: `    "R6": dict(seed=7),` → `    "R1s7": dict(seed=7),   # formerly "R6"`
- line 44: `|R1 - R6|` → `|R1 - R1s7|`

**Cell starting with `# ── Cell 3: Seeding`** (code):
- line 3: `what R6 (seed 7) measures` → `what R1s7 (seed 7, formerly R6) measures`

## 2. `Champion_Task\l2-step1-teacher-rationales.ipynb` (was `l2-stage1-teacher-rationales.ipynb`)

**Cell starting with `# L2 · Stage 1:`** (markdown):
- line 1: `# L2 · Stage 1: teacher answers` → `# L2 · Step 1: teacher answers`

**Cell starting with `# ── Cell 2: Configuration`** (code):
- line 26: `# and Stage 2 falls back` → `# and step 2 falls back`

**Cell starting with `# ── Cell 4: Load manifest, pick the subset`** (code):
- line 58: `Stage 2 falls back to it` → `step 2 falls back to it`

**Cell starting with `# ── Cell 6: Keep only checked-correct rationales`** (code):
- line 4: `in the Stage-2 notebook.` → `in the step 2 notebook.`
- line 6: `# Stage 2 trains that entry` → `# step 2 trains that entry`
- line 45: `Without (Stage 2 trains these` → `Without (step 2 trains these`

## 3. `Champion_Task\r16-find-then-grade.ipynb` (was `r16-two-stage-grader.ipynb`)

**Cell starting with `# R16 · Two-stage model`** (markdown):
- line 1: `# R16 · Two-stage model: find each building, then grade it` → `# R16 · Find-then-grade model: find each building, then grade it`
- line 12: `1. **Stage 1, find buildings:**` → `1. **The finding:**`
- line 13: `2. **Stage 2, grade each building:**` → `2. **The grading:**`
- line 37: `(one row per stage-1 building)` → `(one row per found building)`

**Cell starting with `# ── Cell 2: Configuration and inputs`** (code):
- line 68: `# Stage 1: E3's SAM 3 instance maps` → `# The finding: E3's SAM 3 instance maps`
- line 83: `R16 two-stage model` → `R16 find-then-grade model`
- line 84: `"  Stage 1 (SAM 3): ` → `"  Finding (SAM 3): `

**Cell starting with `## 2. One crop per building`** (markdown):
- line 2: `or from stage 1 (validation)` → `or from the finding (validation)`

**Cell starting with `## 3. Stage 2: the Siamese grader`** (markdown):
- line 1: `## 3. Stage 2: the Siamese grader` → `## 3. The grading: the Siamese grader`

**Cell starting with `# ── Cell 7: Stage 2, the Siamese grader`** (code):
- line 1: `# ── Cell 7: Stage 2, the Siamese grader` → `# ── Cell 7: The grading, the Siamese grader`

**Cell starting with `## 4. Validation: the U-Net`** (markdown):
- line 1: `and the two-stage maps` → `and the find-then-grade maps`
- line 5: `3. **Stage 1:** SAM 3's instances` → `3. **The finding:** SAM 3's instances`

**Cell starting with `# ── Cell 9: Validation, image by image`** (code):
- line 27: `# 3) Stage 1: SAM 3's instances` → `# 3) The finding: SAM 3's instances`
- line 44: `# 4) Grade every stage-1 building` → `# 4) Grade every found building`
- line 72: `"stage1": ...` → **leave** (the column name R16b and R16c step 1 read)
- line 82: `stage-1 buildings graded` → `found buildings graded`

**Cell starting with `# ── Cell 10: Results, the sanity check`** (code):
- line 72: `"variable": "two-stage",` → `"variable": "find-then-grade",`
- line 102: `"# R16 results: two-stage model (find each building, then grade it)"` → `"# R16 results: find-then-grade model (find each building, then grade it)"`
- line 103: `. Stage 1: SAM 3` → `. The finding: SAM 3`
- line 104: `. Stage 2: Siamese ResNet-18 grader,` → `. The grading: Siamese ResNet-18 grader,`
- line 114: `one row per stage-1 building` → `one row per found building`

## 4. `Champion_Task\r16b-grader-prior-correction.ipynb`

**Cell starting with `# R16b · R16's grader corrected`** (markdown):
- line 17: `grades the stage-1 buildings` → `grades the found buildings`
- line 26: `The stage-1 buildings table` → `The found buildings' table`

**Cell starting with `## 2. R16's three maps, rebuilt`** (markdown):
- line 2: `at most one stage-1 building` → `at most one found building`

**Cell starting with `# ── Cell 3: Validation masks`** (code):
- line 25: `rows["stage1"]` → **leave** (column name)
- line 35: `# outside every stage-1 building: background` → `# outside every found building: background`
- line 69: `stage-1 buildings on` → `found buildings on`

**Cell starting with `## 3. The correction, and the new scores`** (markdown):
- line 2: `grades the stage-1 buildings that` → `grades the found buildings that`

**Cell starting with `# ── Cell 4: Corrected grades`** (code):
- line 1: `and the stage-1 buildings' grades` → `and the found buildings' grades`
- line 24: `# Stage-1 buildings that are mostly` → `# Found buildings that are mostly`
- line 43: `Stage-1 buildings that are mostly one true building:` → `Found buildings that are mostly one true building:`
- line 46: `stage-1 buildings by grade:` → `found buildings by grade:`

**Cell starting with `# ── Cell 5: Output files`** (code):
- lines 18 and 19: `"stage1_buildings"`, `"stage1_macro_f1"` → **leave** (keys in its saved `config.json`)
- line 33: `"## Stage-1 buildings that are mostly one true building"` → `"## Found buildings that are mostly one true building"`
- line 34: `stage-1 buildings have at least half` → `found buildings have at least half`
- line 40: `the stage-1 buildings' grades` → `the found buildings' grades`

## 5. `Champion_Task\r16c-step1-ceiling-and-counting.ipynb`

**Cell starting with `# R16c step 1 · R16's ceiling`** (markdown):
- line 7: `Giving each stage-1 building` → `Giving each found building`

**Cell starting with `# ── Cell 1: Configuration and inputs`** (code):
- line 21: `# R16's stage 1: E3's SAM 3 instance maps` → `# R16's finding: E3's SAM 3 instance maps`

**Cell starting with `## 1. The ceiling`** (markdown):
- line 2: `Each stage-1 building is painted` → `Each found building is painted`

**Cell starting with `# ── Cell 3: The ceiling`** (code):
- line 18: `inst["stage1"]` → **leave** (column name)
- line 28: `of each stage-1 building` → `of each found building`
- line 40: `# outside every stage-1 building: background` → `# outside every found building: background`

**Cell starting with `## 2. The counting check`** (markdown):
- line 2: `R16's stage-1 map is rebuilt` → `R16's finding map is rebuilt`

**Cell starting with `# ── Cell 4: Rebuild each image's stage-1 map`** (code):
- line 1: `Rebuild each image's stage-1 map` → `Rebuild each image's finding map`
- line 30: `# R16's stage 1: SAM 3's ids` → `# R16's finding: SAM 3's ids`
- lines 51 and 55: `rows["stage1"]` → **leave** (column name)
- line 73: `R16's stage 1 isn't reproduced` → `R16's finding isn't reproduced`
- line 74: `stage-1 maps; every building matches` → `finding maps; every building matches`

**Cell starting with `# ── Cell 5: Answer the questions`** (code):
- line 56: `"stage1_buildings"` → **leave** (key in its saved `config.json`)
- line 64: `gives each stage-1 building its main` → `gives each found building its main`

## 6. `Champion_Task\r16c-step2-stronger-grader.ipynb`

**Cell starting with `# R16c steps 2 and 3`** (markdown):
- line 1: `# R16c steps 2 and 3 · A stronger R16 grader` → `# R16c step 2 · A stronger R16 grader`
- line 3: `Built as a copy of \`r16-two-stage-grader.ipynb\`` → `Built as a copy of \`r16-find-then-grade.ipynb\``, and in the same line `Stage 1, the R11 reference` → `The finding, the R11 reference`
- line 18: `step 3: SAM 3's outlines` → `step 5 (its own notebook): SAM 3's outlines`
- line 24: replace the whole line with: `- **Step 5 (\`TRAIN_OUTLINES = "sam3"\`)** has its own notebook, \`r16c-step5-grader-on-sam3-outlines.ipynb\`, which needs step 4 (\`r16c-step4-sam3-training-images.ipynb\`) first.`
- line 44: delete the line `- step 3 only: the output of \`r16c-step3a-sam3-training-images\`.`
- line 46: `(step 3: \`R16c-sam3/\`)` → delete those words, so it reads `Outputs in \`/kaggle/working/R16c-truth/\`:`
- line 48: `each stage-1 building` → `each found building`

**Cell starting with `# ── Cell 2: Configuration and inputs`** (code):
- line 22: `Step 3 is this notebook with "sam3", after` → `Step 5 (r16c-step5-grader-on-sam3-outlines.ipynb) is this notebook with "sam3", after`
- line 23: `# r16c-step3a-sam3-training-images.ipynb has run` → `# r16c-step4-sam3-training-images.ipynb (step 4) has run`
- line 25: `"sam3" (step 3)` → `"sam3" (step 5)`
- line 82: `# Stage 1 on the validation images:` → `# The finding on the validation images:`
- line 84: `"add the output of r16c-step3a-sam3-training-images"` → `"add the output of r16c-step4-sam3-training-images"`
- line 92: `'SAM 3 outlines (step 3)'` → `'SAM 3 outlines (step 5)'`
- line 95: `"  Stage 1 (SAM 3): ` → `"  Finding (SAM 3): `

**Cell starting with `# ── Cell 4: Validation split`** (code):
- line 14: `finish step 3a"` → `finish step 4"`

**Cell starting with `## 2. One crop per building`** (markdown):
- line 4: `- **Step 3 (\`sam3\`):**` → `- **Step 5 (\`sam3\`, its own notebook):**`

**Cell starting with `# ── Cell 5: Crop helpers`** (code):
- line 52: `"None" (step 3 only)."""` → `"None" (step 5 only)."""`

**Cell starting with `## 3. Stage 2: the Siamese grader, one per seed`** (markdown):
- line 1: `## 3. Stage 2: the Siamese grader` → `## 3. The grading: the Siamese grader`

**Cell starting with `# ── Cell 7: Stage 2, the Siamese grader per seed`** (code):
- line 1: `# ── Cell 7: Stage 2, the Siamese grader` → `# ── Cell 7: The grading, the Siamese grader`

**Cell starting with `## 4. Validation, once`** (markdown):
- line 5: `3. **Stage 1 as in R16:**` → `3. **The finding, as in R16:**`

**Cell starting with `# ── Cell 9: Validation, image by image`** (code):
- line 32: `# 3) Stage 1 as in R16:` → `# 3) The finding, as in R16:`
- line 91: `"stage1": ...` → **leave** (column name in step 2's saved table)
- line 102: `stage-1 buildings graded` → `found buildings graded`

**Cell starting with `# ── Cell 10: Results, the sanity check`** (code):
- line 90: `"variable": "two-stage",` → `"variable": "find-then-grade",`
- line 135: `. Stage 1 as in R16: SAM 3` → `. The finding as in R16: SAM 3`
- line 141: `each stage-1 building` → `each found building`
- line 155: `one row per stage-1 building` → `one row per found building`

## 7. `Champion_Task\e3-sam3-building-finding.ipynb`

**Cell starting with `# E3 · SAM 3 smoke test`** (markdown):
- line 6: `can be stage 1 of the two-stage model R16` → `can do the finding for R16, the find-then-grade model`
- line 18: `For R16, a stage 1 must find the footprints` → `For R16, the finding must find the footprints`
- line 35: `R16 uses SAM 3 as stage 1.` → `R16 uses SAM 3 for the finding.`

**Cell starting with `# ── Cell 7: Summary, comparison with R9`** (code):
- line 28: `that stage 1 would hand to stage 2.` → `that the finding would hand to the grading.`
- line 36: `stage1 = ...` → **leave** (a variable name)
- line 39: `R16 stage 1 = {stage1}.` → `R16's finding = {stage1}.` (change only the text before `{stage1}`)

---

## What stays as it is

- **Recorded outputs and saved results files** (printed text, `README.md` and CSV files written by these runs) keep the old words, as records of what ran. The changes above make any re-run write the new words.
- **Kaggle names:** you renamed `r16-two-stage-grader` and `l2-stage1-teacher-rationales` on Kaggle too (20 Oct). R6's Kaggle notebook is renamed when you find it. `r16c-step2-stronger-grader` keeps its name, and its output folder stays `R16c-truth`, because that is written by the run and step 3 looks for it.
- **Hugging Face:** the R6 run's checkpoint repo stays `AbrarAlam/disasterm3-d4-R6`.
- **Say when you're done**, so the files can be re-checked by searching for the old names.
