import os
from PIL import Image

def resize_images(folder_path, size=(512, 512)):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            filepath = os.path.join(folder_path, filename)
            try:
                img = Image.open(filepath)
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(filepath)
                print(f"Resized {filename} to {size}")
            except Exception as e:
                print(f"Error resizing {filename}: {e}")

if __name__ == "__main__":
    target_folder = "data/images/00000000"
    if os.path.exists(target_folder):
        resize_images(target_folder)
        print("Done resizing images!")
    else:
        print(f"Folder {target_folder} not found.")
