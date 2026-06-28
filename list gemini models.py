import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load the environment variables from your specific path
env_path = r"D:\CSE499AB_project\.env"
load_dotenv(dotenv_path=env_path)

# 2. Retrieve the API key 
# (Update "GEMINI_API_KEY" if you named your variable differently in the .env file)
api_key = os.getenv("GEMINI_API_KEY") 

if not api_key:
    print(f"Error: Could not find 'GEMINI_API_KEY' in {env_path}")
    print("Please ensure your .env file is formatted correctly (e.g., GEMINI_API_KEY=your_key_here).")
else:
    # 3. Configure the Gemini client with the extracted key
    genai.configure(api_key=api_key)

    print("Fetching available Gemini models...\n")
    
    # 4. Iterate through and list the models
    for m in genai.list_models():
        # Optional: Filter to only show models that support text/chat generation
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
            print(f"Description: {m.description}")
            print("-" * 40)