# E5 · Prompt for Ridita: collect the leak and BRIGHT audit files

> **Answered 26 Oct.** Ridita sent `audit_outputs/README.md` and `val_train_leaks.csv`. The README confirms the 40 leaked pairs and lists the 19 that Abrar's check missed. Her CSV matches the combined list, [leaked_pairs_40.csv](../Champion_Task/E5-results/E5/leaked_pairs_40.csv), exactly. ~~Still open: push the `audit_outputs/` folder.~~ Pushed 26 Oct (commit `74d44059e`), checked against what she sent. E5's own steps 1 and 2 then confirmed the 40 a third time.

**Why:** E5 needs the exact list of leaked validation pairs, `val_train_leaks.csv`, to re-score O1 on clean images. Ridita's corrected audit names that file, but it isn't in the repo. Abrar's local check found 21 leaked pairs; the audit says 40. Details: [E5-results/E5/README.md](../Champion_Task/E5-results/E5/README.md).

**How Ridita runs it:**
1. `git pull`. E5's files have been on GitHub since 26 Oct (commit `5ff744c`).
2. Open the repo in the IDE and paste the block below into the coding assistant's chat, from the repo root.
3. Send Abrar `val_train_leaks.csv` as soon as step 2 of the prompt confirms it, then the summary from step 6.

```text
I wrote the DisasterM3 vs BRIGHT optical audit for our CSE499B project. Abrar needs everything behind it, above all the train/validation leak, so he can re-score our O1 results on clean images. Collect it, check it and package it in the repo. Don't change, move or delete any original file; copy only.

Read these first:
- My report: @Essential_task_delegation/BRIGHT_DisasterM3_Integration/DisasterM3_vs_BRIGHT_Optical_Audit_EDA_corrected.pdf (§8 is the leak, §11 lists the scripts and output files).
- Abrar's independent check of the leak: @Essential_task_delegation/Champion_Task/E5-results/E5/README.md and @Essential_task_delegation/Champion_Task/E5-results/E5/leaked_pairs_local_check.csv. Abrar found 21 leaked pairs by hashing xBD's original before-images; my report says 40.
- How our 645 validation pairs are defined: @Essential_task_delegation/Champion_Task/d4-champion-model-R11.ipynb. "Cell 4: Load manifest and extract segmentation entries" keeps the optical pairs from train_release.json, and Cell 5 splits them with train_test_split(pairs, test_size=0.1, random_state=42).

1. Find these files on my computer and list each one's full path, size, date and row count. Start in Essential_task_delegation/BRIGHT_DisasterM3_Integration/ and its data/dm3_bright_optical_eda/ folder, then search the rest of the disk.
   - Scripts: dm3_bright_optical_eda.py, bright_dm3_dedup.py, dm3_bright_audit.py, dm3_vs_bright_compare.py, and bright_disasterm3_standardizer.py (deleted on 25 Oct; it's in git history, commit 922dbd2b4).
   - Outputs: summary.json, dm3_images.csv, bright_pre_tiles.csv, bright_targets.csv, dm3_pool_labels.csv, dm3_optical_pair_registration.csv, dm3_pool_pairs_radiometry.csv, dm3_events.csv, val_train_leaks.csv, bright_dm3_dedup_report.json.
   - The HTML or Markdown source the PDF was printed from, and its figures.
   If something is missing, say so and give the command that would make it again.

2. Check the leak, and show your working.
   - val_train_leaks.csv: its columns, its row count (my report says 40), and the destroyed pixels in those validation pairs (155,812 of 1,808,111).
   - Prove my 645 validation pairs are the same as the notebook's: the same pool rule, the same pair order before the split, the same random_state. Report the total destroyed pixels in validation (it must be 1,808,111) and the number of socal_fire and woolsey_fire validation pairs (Abrar's split has 54 and 32).
   - Say exactly what was hashed to find the twins (file bytes or decoded pixels, and from which folder), and how many before-images socal_fire and woolsey_fire share inside the pool (my report says 264).
   - Compare with Abrar's 21: which of his 21 are in my 40, and for each of my other pairs, why his check on xBD's original files missed it (for example, DisasterM3's copies are identical where xBD's originals differ). Also count validation pairs whose twin is also in validation (not a leak), and twins that aren't in the pool.

3. Copy everything into a new folder, Essential_task_delegation/BRIGHT_DisasterM3_Integration/audit_outputs/, with scripts/ and outputs/ inside.
   - Leave out images, masks, model files, anything over 50 MB, and Offline_RAG_KR2/guidelines/Sphere_Handbook_2018_EN.pdf.
   - Check every copied file for tokens, passwords or API keys, and list any absolute paths from my computer that appear inside the CSVs.

4. Write audit_outputs/README.md with:
   - what each file is, and which script makes it;
   - how to make every number again: the commands, the inputs each one needs (DisasterM3 Instruct images and train_release.json, the BRIGHT tiles) and where to get them, and the run time;
   - the leak method and numbers, and the comparison with Abrar's 21;
   - why the four files in BRIGHT_DisasterM3_Integration/ were deleted on 25 Oct (commits 3ae5ab0ca, 26ab6fb4a, 922dbd2b4, a33aa8b12), and what replaced them.

5. Optional, only if DisasterM3's benchmark images are on this computer: run the same twin check between the benchmark (Bench) and Instruct. It decides whether our 7B's benchmark scores are affected too. If they aren't here, say so; Abrar can run it on Kaggle.

6. Don't commit or push. At the end, give me:
   - a summary with the numbers and the paths of the new files;
   - the git commands to add only audit_outputs/, and to take Sphere_Handbook_2018_EN.pdf out of the public repo (git rm --cached, plus a .gitignore line), with a commit message that has no AI or assistant attribution.
```
