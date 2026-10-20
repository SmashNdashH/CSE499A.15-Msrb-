# Tamanna's Action Plan: Architecture Ablation Lead

Hello Tamanna! As Member B, your mission is to run a controlled ablation study on the segmentation architectures. This means you will test 6 different model architectures while keeping all other variables (like loss function, data augmentation, batch size, and epochs) exactly the same. 

Here is your step-by-step guide to executing this task:

### Phase 1: Setup on Kaggle
1. **Find the Baseline Notebook**: Locate `segmentation-model (1).ipynb` on Kaggle (or upload it if you have it locally).
2. **Fork the Notebook**: Create 6 separate copies (forks) of this notebook in your Kaggle workspace. 
3. **Rename the Notebooks**: Name them clearly so you don't mix them up:
   - `segmodel_ablation_B1_unet.ipynb`
   - `segmodel_ablation_B2_unetpp.ipynb`
   - `segmodel_ablation_B3_unetpp_scse.ipynb`
   - `segmodel_ablation_B4_deeplabv3plus.ipynb`
   - `segmodel_ablation_B5_unet_resnet50.ipynb`
   - `segmodel_ablation_B6_manet.ipynb`

### Phase 2: Modify the Notebooks
For each notebook, you must **ONLY** change the model definition (which should be in Cell 8) and the experiment ID for saving results.
- Open each notebook.
- Go to Cell 8.
- Replace the existing `model = smp.Unet(...)` code with the corresponding code for that experiment. (I have provided all the exact code snippets in `Model_Definitions.py` in this folder).
- Go to the Evaluation cell (Cell 9) and ensure the CSV saving script uses the correct `experiment_id` and `value` (I have provided the template in `Evaluation_Snippet.py`).

### Phase 3: Train and Checkpoint
- Run each notebook on a Kaggle T4 GPU.
- **Important**: Make sure you have the checkpointing code in your training loop so that you save `best_model_MEMBER_B_EXP_X.pth` at each epoch if the mIoU improves. This protects you from Kaggle timeouts.
- Let each model train for the full 40 epochs. *Note: For DeepLabV3+ and MA-Net, if you get an Out of Memory (OOM) error, reduce the batch size to 8.*

### Phase 4: Collect Results
- After each notebook finishes, a CSV file will be generated (e.g., `B2_arch_unetpp.csv`).
- Download the CSV file from Kaggle.
- Fill in the values in the `Deliverables_Report.md` file (which I have also created for you in this folder).
- Write a short 1-paragraph conclusion at the bottom of the report explaining which model performed best and why.

---
**Remember the Golden Rule:** Do not change the loss function, the image sizes, or the data augmentations. If you change more than one thing, the ablation study becomes invalid!
