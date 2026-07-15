import os
import json
import re
from openai import OpenAI  # kept for dependency reference if needed but unused
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
from crewai.tools import tool
from dotenv import load_dotenv
import base64
import io

load_dotenv()

# Set up API key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
genai.configure(api_key=api_key)

def encode_image(image_path, target_size):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
    with io.BytesIO() as buffer:
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_images_in_folder(folder_path,target_size=(256,256)):
    encoded_image = []
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg') ]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        encoded_image.append(encode_image(image_path, target_size))
    return encoded_image

script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "Disaster_image", "Global_image")
encoded_images = encode_images_in_folder(folder_path, target_size=(1080, 1522)) #encoded_images[i] #1000x1000


def annotate_image(image_path, annotations, output_path, size=None):
    image = Image.open(image_path)
    if size is not None:
        image = image.resize(size)
    draw = ImageDraw.Draw(image)

    # Define font
    font_size = 150
    font_color = (255, 255, 0)  # Red color
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw annotations on the image
    for annotation in annotations:
        position, text = annotation['position'], annotation['text']
        draw.text(position, text, font=font, fill=font_color)

    # Save the annotated image
    image.save(output_path)


model = genai.GenerativeModel("gemini-3.1-flash-lite")


@tool
def global_map_annotation(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The input from the LLM.
    """

    base_instruction = (
        "these images are the same place that experienced disaster. "
        "There are different disaster grade such as G1~G10 for the different locations in these images. "
        "Firstly, please find the following locations in the first image according to location names; Secondly, "
        "generate a json structure for all the relevant disaster locations with position and grading information "
        "to annotate these labels in the second image like this: "
        '{"annotations": [{"position": [820, 380], "text": "G1"}, {"position": [660, 620], "text": "G2"}]}. '
        "The disaster locations with relevant grading are following: "
    )
    prompt = base_instruction + text
    print(prompt)

    contents = [prompt]
    for image in encoded_images:
        contents.append({
            "mime_type": "image/png",
            "data": image
        })

    import time
    llm_response = None
    for attempt in range(5):
        try:
            response = model.generate_content(contents)
            llm_response = response.text
            break
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                print(f"Rate limit hit in global_map_annotation, retrying in 15 seconds... (Attempt {attempt+1}/5)")
                time.sleep(15)
            else:
                return f"An error occurred: {str(e)}"
    
    if llm_response is None:
        return "Failed after multiple attempts due to rate limit."

    try:
        print(llm_response)

        json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)

        content_dict = None
        if json_match:
            json_str = json_match.group(1).strip()
            json_str = re.sub(r'//.*', '', json_str)
            try:
                content_dict = json.loads(json_str)
                print(content_dict['annotations'])
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
        else:
            try:
                content_dict = json.loads(llm_response.strip())
                print(content_dict['annotations'])
            except json.JSONDecodeError as e:
                print("No JSON structure found in the content.")

        if content_dict and 'annotations' in content_dict:
            image_path = os.path.join(script_dir, "Disaster_image", "Global_image", "2.jpg")
            output_path = os.path.join(script_dir, "Disaster_image", "Global_image", "4_anotated.png")
            annotate_image(image_path, content_dict['annotations'], output_path, size=(1080, 1522))
        else:
            print("Could not annotate image: invalid or missing annotations structure.")

        return text

    except Exception as e:
        return f"An error occurred: {str(e)}"