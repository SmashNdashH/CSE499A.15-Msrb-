import os
import ollama
import argparse #allow for command line arugments to select a model 

# --- Configuration ---
# Updated with generic relative paths for GitHub/Poster consistency
BASE_RESULTS_DIR = "./data/images/"

def parsing_arguments():
    parser = argparse.ArgumentParser(description="Run VLM damage assessment on image pairs.")

    parser.add_argument(
            "--model", "-m",
            type=str,
            default = "gemma3:27b",
            help = "The name of the VLM to use (Default is Gemma3:27b)"
            )

    return parser.parse_args()


def run_assessment(model):

    # Generic output directory structure
    base_output = "./results/vlm_assessments/"
    model_safe_name = model.replace(':', '_')
    final_output_location = os.path.join(base_output, model_safe_name)
    os.makedirs(final_output_location, exist_ok=True)

    print(f"Results are being saved to: {final_output_location}")

    for i in range(0, 2): 
        # Format the ID as 8 digits (e.g., 00000123)
        folder_id = f"{i:08d}"
        
        # Construct the folder path and generic filenames
        folder_path = os.path.join(BASE_RESULTS_DIR, folder_id)
        pre_filename = f"{folder_id}_pre_disaster.png"
        post_filename = f"{folder_id}_post_disaster.png"
        
        pre_path = os.path.join(folder_path, pre_filename)
        post_path = os.path.join(folder_path, post_filename)

        # Check if both images actually exist before calling the model
        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            print(f"Skipping {folder_id}: Files not found in {folder_path}")
            continue

        print(f"\n{'='*20} Processing Folder: {folder_id} {'='*20}")
        
        full_response_text = ""

        try:
            # Start the stream with the two specific images
            model_stream = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': f"You are an expert damage assessment analyzer. Your job is to take images from before and after a natural disaster has happened and analyze the structures of interest. The structures of interest you are to analyze are buildings, both residential and non-residential. You are going to classify each building inside the image on a scale of 1-4. 1 being no/slight damage - thus being of least concern, 2 being moderate damage thus being of moderate concern, 3 being major damage thus being of severe concern, 4 being totally collapsed thus having the most concern. You are to take in factors such as hazards to the disaster recovery experts, hazards to civilians that may be trapped, hazards to future damages if a building collapses, and any other hazards that pose a risk to humans or future buildings. In your analysis of each building you are also to write a description of the building and why you classified it the way you did into the one of 4 categories above. The information you provide will be given to disaster recovery experts, workers, and analyzers to help secure the area after a disaster has occurred. Be absolutely honest in your assessment in what you see to provide the most accurate and safe results for recovery workers. Remember that some post disaster imagery won't have any damages visible, in such cases you should tell the user that. To the best of your ability provide the most accurate damage assessment you can to provide the best help and reduce increased damages and casualties. Differences in seasons, time of day, and other natural and non-destructive causes similar to those should be taken into account helping create a more accurate damage assessment. Analyze these two images where image 1 is Pre-Disaster and image 2 is Post-Disaster.",
                    'images': [pre_path, post_path]
                }],
                options={
                    'num_ctx': 4096, # reduced context window to prevent memory/timeout issues
                    },
                stream=True, 
            )

            # Print to console in real-time
            for chunk in model_stream:
                content = chunk['message']['content']
                print(content, end='', flush=True)
                full_response_text += content

            # Save to its own text file
            output_file = os.path.join(final_output_location, f"{folder_id}_assessment.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_response_text)
            
            print(f"\n\n[SUCCESS] Saved assessment to: {output_file}")

        except Exception as e:
            print(f"\n[ERROR] Failed to process {folder_id}: {e}")

if __name__ == "__main__":
    args = parsing_arguments()
    run_assessment(args.model)
    print("\nAll tasks complete.")
