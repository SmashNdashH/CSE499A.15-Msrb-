# Copy and paste this at the end of the evaluation cell (Cell 9) in Kaggle

import csv
import time

# NOTE: Update the 'experiment_id' and 'value' fields for each notebook!
# Examples: 
# For B1: "experiment_id": "B1_arch_unet_baseline", "value": "Unet_resnet34"
# For B2: "experiment_id": "B2_arch_unetpp", "value": "UnetPlusPlus_resnet34"
# For B3: "experiment_id": "B3_arch_unetpp_scse", "value": "UnetPlusPlus_scse_resnet34"
# For B4: "experiment_id": "B4_arch_deeplabv3plus", "value": "DeepLabV3Plus_resnet34"
# For B5: "experiment_id": "B5_arch_unet_resnet50", "value": "Unet_resnet50"
# For B6: "experiment_id": "B6_arch_manet", "value": "MANet_resnet34"

results = {
    "experiment_id": "B2_arch_unetpp",   # <-- UPDATE THIS
    "variable": "architecture",
    "value": "UnetPlusPlus_resnet34",    # <-- UPDATE THIS
    
    # Assuming 'metrics' dictionary and 'epoch_times' list exist in your evaluation code
    "miou": float(metrics["iou"].mean()),
    "bg_iou": float(metrics["iou"][0]),
    "intact_iou": float(metrics["iou"][1]),
    "damaged_iou": float(metrics["iou"][2]),
    "destroyed_iou": float(metrics["iou"][3]),
    "precision": float(metrics["precision"].mean()),
    "recall": float(metrics["recall"].mean()),
    "f1": float(metrics["f1"].mean()),
    
    # Calculate training time in hours
    "train_time_hrs": sum(epoch_times) / 3600 if 'epoch_times' in locals() else 0.0,
    "notes": "Batch size 16", # Change to 8 if you had to reduce it
}

file_path = f"/kaggle/working/{results['experiment_id']}.csv"

with open(file_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=results.keys())
    w.writeheader()
    w.writerow(results)

print(f"Results successfully saved to {file_path}")
