# Weekly Progress Report: Code implementation

## Overview
This week focused on setting up the local environment, optimizing model selection for Vision-Language Models (VLMs), and testing the damage assessment pipeline. 

## Important Terminologies
- **Vision-Language Model (VLM):** An AI model capable of processing and understanding both visual (images) and textual data simultaneously.
- **LLaVA:** (Large Language-and-Vision Assistant) The specific VLM chosen for this project to interpret disaster images and generate descriptions.
- **Ollama:** A framework used to run large language and vision models locally on a personal machine without relying on cloud APIs.
- **Inference Time:** The time it takes for the model to process the images and generate the assessment output. (A major challenge encountered this week).
- **Pre/Post-Disaster Imagery:** The paired dataset of images showing a location before and after a natural disaster, used as input for the model.
- **Damage Assessment Pipeline:** The end-to-end process of feeding images into the model and extracting a structured evaluation of structural damage.

## Key Activities

### 1. Local Environment Setup
- Successfully installed **Ollama** on the local machine to enable running VLMs offline for image analysis.

### 2. Model Selection & Performance Tuning
- Encountered significant performance bottlenecks (huge **inference times**) during initial testing with Ollama.
- Researched and transitioned to the **LLaVA** model to try and improve inference times. 
- Discovered that while LLaVA is highly capable, inference times remain high due to local hardware constraints when processing image pairs.

### 3. Codebase Updates & Synchronization
- Synchronized the local workspace with the remote repository by pulling the latest changes (`git pull`) to ensure the implementation was using the most up-to-date scripts.

### 4. Testing & Debugging the Pipeline
- Investigated the `VLM_Damage_Assessment.py` script to understand its scope, noting it was hardcoded to process 227 pre/post-disaster image pairs.
- To work around the high inference times and allow for rapid testing, iteratively modified the codebase to process smaller subsets of data:
  - First scaled down to **10 data points**.
  - Further reduced to **5 data points**.
  - Finalized testing on just **2 data points** for immediate validation.

## How the Pipeline Works
1. **Input Data Loading:** The script iterates through a specific set of folders (e.g., `00000000`, `00000001`) inside the `./data/images/` directory.
2. **Image Pairing:** For each folder, it loads two corresponding images: a `pre_disaster.png` and a `post_disaster.png`.
3. **Prompting the VLM:** Both images are sent to the **LLaVA** model running locally via Ollama, along with a detailed prompt asking the model to act as a damage assessment expert and categorize the damage on a scale of 1-4.
4. **Generating Assessment:** The model streams its analysis, describing the condition of the buildings (e.g., minor damage, total collapse) and noting any potential hazards.

## Outputs Saved
- **Format:** The model's analysis is saved as a detailed text report (`.txt` format).
- **Location:** The results are dynamically saved into a model-specific directory: `./results/vlm_assessments/llava/`.
- **Naming Convention:** Each output file is named according to its corresponding data point (e.g., `00000000_assessment.txt`).

## Next Steps
- Continue evaluating strategies to mitigate high model running times (e.g., batching, resizing images, or utilizing cloud compute if local hardware is insufficient).
- Scale up the analysis to the full dataset once performance is optimized.
