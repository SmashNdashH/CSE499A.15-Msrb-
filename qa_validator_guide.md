# 🛰 VLM Dataset QA Validator - Team Guide

Welcome to the data validation phase! We are using a custom Human-in-the-Loop QA tool (`dataset_validator.py`) to review, edit, and approve the generated image captions for our Vision-Language Model. 

This guide covers everything you need to start validating the dataset.

## 📋 Prerequisites & Setup

It is highly recommended to use a virtual environment so you don't conflict with your system-wide Python packages.

### 1. Create and Activate a Virtual Environment
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Once your virtual environment is active, install the project dependencies using the `requirements.txt` file at the root of the project:
```bash
pip install -r requirements.txt
```
*(If you are only running the validator and don't need the other ML libraries, you can alternatively just run `pip install customtkinter pillow`)*

## 🚀 How to Run the Tool
Make sure you are at the root of the project (`disaster-vlm-project/`), and then run:
```bash
python support/dataset_validator.py
```

> [!NOTE]
> The tool is designed to be completely safe to close at any time. It automatically tracks your progress and will resume exactly where you left off the next time you open it!

## 🎮 Controls & Workflow

Once the UI opens, you will see the satellite image on the left and the **Editable Rescue Report** on the right. 

### Core Actions
- **✓ Accept (Right Arrow `→`)**: Saves the current sample to the verified dataset. **If you edited the text, your edits are saved!**
- **✗ Reject (Left Arrow `←`)**: Discards the sample. It will not be included in the final dataset.
- **↩ Undo (Up Arrow `↑`)**: Steps back to the previous sample so you can review it again. (Note: If you re-accept an undone sample, it will cleanly overwrite the old entry without creating duplicates).

### Image Navigation
- **Zoom**: Use your **Mouse Scroll Wheel** to zoom in and out of the image (it centres on your pointer).
- **Pan**: **Click and Drag** on the image to move around when zoomed in.

## 📂 Where Does the Data Go?
- **Input**: The tool reads from `dataset/train_dataset.jsonl`
- **Output**: When you hit "Accept", the sample is appended to `dataset/verified_dataset.jsonl`. 

> [!IMPORTANT]
> Do not manually edit `dataset/verified_dataset.jsonl` while the validator app is running, as this can cause sync issues. 

Happy validating! If you encounter any unexpected bugs, let the team know.
