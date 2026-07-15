import os
import base64
import io
from PIL import Image

from crewai.tools import tool
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Set up API key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
genai.configure(api_key=api_key)

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "Disaster_image", "Global_image")

def encode_image(image_path, target_size):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
    with io.BytesIO() as buffer:
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_images_in_folder(folder_path,target_size=(256,256)):
    encoded_image = []
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        encoded_image.append(encode_image(image_path, target_size))
    return encoded_image


encoded_images = encode_images_in_folder(folder_path, target_size=(720,720)) #encoded_images[i]

model = genai.GenerativeModel("gemini-3.1-flash-lite")

@tool
def global_img_interpreter(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The input from the LLM.
    """
    prompt = "Analyze the map for potential dangerous areas. " + text
    print("Global information:", prompt)

    contents = [prompt]
    for image in encoded_images:
        contents.append({
            "mime_type": "image/png",
            "data": image
        })

    import time
    for attempt in range(5):
        try:
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                print(f"Rate limit hit, retrying in 60 seconds... (Attempt {attempt + 1}/5)")
                time.sleep(60) 
            else:
                return f"An error occurred: {str(e)}"
    return "Failed after multiple attempts due to rate limit."




