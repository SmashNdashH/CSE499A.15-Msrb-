import json
import os
import re

def main():
    # Setup paths relative to the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bench_file = os.path.join(current_dir, "benchmark_release.json")
    results_dir = os.path.join(current_dir, "DisasterM3 Evaluation Results")
    safe_model = "AbrarAlam--disasterm3-qwen2.5vl7b-mergedFP"
    
    # 1. Check if the user has downloaded the benchmark file locally
    if not os.path.exists(bench_file):
        print("❌ ERROR: Missing 'benchmark_release.json'!")
        print(f"Please download it from your Kaggle Dataset and place it here:\n{bench_file}")
        return

    print(f"Loading ground truth from {bench_file}...")
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    # 2. Build ground truth lookup map
    task_map = {
        "Disaster Bearing Bodies Recognition": "bearing_body",
        "Building Damage Counting": "building_damage_counting",
        "Disaster Type Recognition": "disaster_type",
        "Road Damage Counting": "road_damage_counting",
        "Disaster Scene Recognition": "landuse",
        "Relational Reasoning": "relational_reasoning_qa"
    }

    gt = {}  
    for entry in bench_data:
        task = entry.get("task")
        if task not in task_map:
            continue
        subset = task_map[task]
        if subset not in gt:
            gt[subset] = []
        gt[subset].append(entry.get("ground_truth_option", ""))

    gt_by_id = {}
    for subset, answers in gt.items():
        for idx, ans in enumerate(answers):
            if isinstance(ans, list):
                gt_by_id[f"{subset}_{idx}"] = ", ".join(str(a) for a in ans)
            else:
                gt_by_id[f"{subset}_{idx}"] = str(ans)

    # 3. Score results
    print("\n" + "=" * 75)
    print(f"{'Task (Subset)':<30} | {'Status':>13} | {'Total':>8} | {'Correct':>7} | {'Acc%':>7}")
    print("=" * 75)

    for subset_name in ["bearing_body", "building_damage_counting", "disaster_type",
                        "road_damage_counting", "landuse", "relational_reasoning_qa"]:
        
        jsonl_path = os.path.join(results_dir, subset_name, safe_model, "finished.jsonl")
        total_in_benchmark = len([k for k in gt_by_id if k.startswith(subset_name + "_")])
        
        if not os.path.exists(jsonl_path):
            print(f"{subset_name:<30} | {'---':>13} | {total_in_benchmark:>8} | {'---':>7} | {'N/A':>7}")
            continue
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            preds = [json.loads(line.strip()) for line in f if line.strip()]
        
        correct = 0
        for pred in preds:
            pred_id = pred["id"]
            pred_response = str(pred["response"]).strip()
            gt_answer = str(gt_by_id.get(pred_id, ""))
            
            # Normalize: robustly extract only the option letters (A-Z) to ignore periods/brackets
            pred_set = set(re.findall(r'[A-Z]', pred_response.upper()))
            gt_set = set(re.findall(r'[A-Z]', gt_answer.upper()))
            
            if pred_set == gt_set:
                correct += 1
        
        acc = (correct / len(preds) * 100) if len(preds) > 0 else 0
        status = "COMPLETE" if len(preds) == total_in_benchmark else f"INC: {len(preds)}/{total_in_benchmark}"
        
        print(f"{subset_name:<30} | {status:>13} | {total_in_benchmark:>8} | {correct:>7} | {acc:>6.2f}%")

    print("=" * 75)

if __name__ == "__main__":
    main()
