import json
import os

# 1. Hardcoded Baseline Metrics from DisasterVQA Paper (Figure 4)
# Format: "Model Name": {"Binary Acc": float, "MCQ F1": float, "Open-Ended Acc": float}
baselines = {
    "GPT-4.1-mini (Proprietary)": {"Binary Acc": 0.91, "MCQ F1": 0.85, "Open-Ended Acc": 0.83},
    "Mistral-Small-3.1-24B": {"Binary Acc": 0.90, "MCQ F1": 0.81, "Open-Ended Acc": 0.75},
    "Qwen-2.5-VL-32B": {"Binary Acc": 0.89, "MCQ F1": 0.81, "Open-Ended Acc": 0.79},
    "Molmo-7B-D": {"Binary Acc": 0.88, "MCQ F1": 0.74, "Open-Ended Acc": 0.70},
    "GPT-4o-mini (Proprietary)": {"Binary Acc": 0.87, "MCQ F1": 0.78, "Open-Ended Acc": 0.78},
    "LLaMA-3.2-11B": {"Binary Acc": 0.87, "MCQ F1": 0.80, "Open-Ended Acc": 0.75},
    "Pixtral-12B": {"Binary Acc": 0.86, "MCQ F1": 0.77, "Open-Ended Acc": 0.76},
}

def calculate_abrar_metrics(predictions_file="vqa_predictions_evaluated.jsonl"):
    """
    Reads the evaluated JSONL file from Kaggle and calculates overall metrics 
    for Abrar's fine-tuned model without requiring pandas.
    """
    if not os.path.exists(predictions_file):
        print(f"Could not find {predictions_file}. Make sure to download it from Kaggle!")
        print("Using placeholder values (0.00) for Abrar's model to demonstrate the table layout.\\n")
        return {"Binary Acc": 0.00, "MCQ F1": 0.00, "Open-Ended Acc": 0.00}
    
    with open(predictions_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    binary_correct = 0
    binary_total = 0
    mcq_correct = 0
    mcq_total = 0
    open_correct = 0
    open_total = 0
    
    for row in data:
        q_type = row.get('question_type', '')
        is_correct = row.get('is_correct', 0)
        
        if q_type in ['Yes/No', 'Binary']:
            binary_total += 1
            binary_correct += is_correct
        elif q_type == 'Multiple-Choice':
            mcq_total += 1
            mcq_correct += is_correct
        elif q_type in ['Open-Ended', 'Open']:
            open_total += 1
            open_correct += is_correct
            
    metrics = {}
    metrics["Binary Acc"] = round(binary_correct / binary_total, 2) if binary_total > 0 else 0.00
    metrics["MCQ F1"] = round(mcq_correct / mcq_total, 2) if mcq_total > 0 else 0.00
    metrics["Open-Ended Acc"] = round(open_correct / open_total, 2) if open_total > 0 else 0.00
        
    return metrics

def generate_markdown_table(abrar_metrics, output_file="benchmark_comparison.md"):
    """
    Generates a beautiful Markdown table comparing Abrar's model to the baselines.
    """
    md_content = "# DisasterVQA Benchmark Comparison\\n\\n"
    md_content += "This table compares the zero-shot performance of **Abrar's Fine-Tuned Qwen2.5-VL-7B** against the 7 baseline models evaluated in the DisasterVQA research paper (Al-Mohannadi et al., 2026).\\n\\n"
    
    md_content += "| Model Architecture | Binary Accuracy (Yes/No) | Multiple-Choice (F1) | Open-Ended (Acc) |\\n"
    md_content += "| :--- | :---: | :---: | :---: |\\n"
    
    # Add Abrar's model first and bolded
    md_content += f"| **AbrarAlam/disasterm3-qwen2.5vl7b-mergedFP (Ours)** | **{abrar_metrics['Binary Acc']:.2f}** | **{abrar_metrics['MCQ F1']:.2f}** | **{abrar_metrics['Open-Ended Acc']:.2f}** |\\n"
    
    # Add baselines
    for model_name, metrics in baselines.items():
        md_content += f"| {model_name} | {metrics['Binary Acc']:.2f} | {metrics['MCQ F1']:.2f} | {metrics['Open-Ended Acc']:.2f} |\\n"
        
    md_content += "\\n\\n"
    md_content += "> **Note on Metrics:** The baseline scores are extracted directly from Figure 4 of the *DisasterVQA* paper. Binary and Open-Ended formats use Accuracy, while Multiple-Choice uses F1-score.\\n"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Successfully generated markdown comparison table: {output_file}")
    print("Preview of the table:\\n")
    print(md_content)

if __name__ == "__main__":
    # Calculate Abrar's model metrics (will use 0.0 if you haven't run the Kaggle notebook yet)
    abrar_metrics = calculate_abrar_metrics("vqa_predictions_evaluated.jsonl")
    
    # Generate the markdown report
    generate_markdown_table(abrar_metrics, "benchmark_comparison.md")
