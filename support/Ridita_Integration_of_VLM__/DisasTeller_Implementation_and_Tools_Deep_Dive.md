# DisasTeller Implementation Overview

Ridita implemented **DisasTeller**, an autonomous multi-agent disaster management simulation system. It uses **CrewAI** to orchestrate several AI agents powered by the **Gemini 3.1 Flash-Lite** Large Language Model (and Vision models). The system mimics how different teams (Experts, Alerts, Emergency, and Assignment) would collaborate to assess a disaster, alert the public, allocate resources, and plan reconstruction by processing multi-modal data (images, PDFs, and internet searches).

There are no Jupyter notebooks in the target directories; the implementation is purely script-based. Below is a detailed script-by-script breakdown of the implementation, complete with relevant code snippets.

---

## What is CrewAI and How is it Implemented Here?

**CrewAI** is an open-source framework designed to orchestrate autonomous AI agents. Instead of having a single AI try to solve a massive problem all at once, CrewAI lets you create a "crew" of specialized AI agents. You give each agent a specific role, a backstory, and a set of tools, and they work together (like a human team) to complete complex, multi-step workflows.

Here is exactly how Ridita implemented the core concepts of CrewAI in her script (`DisasterManagement_teamwork_simulation.py`):

### 1. Agents (The Employees)
An Agent in CrewAI is an autonomous AI with a specific persona. Ridita created four distinct agents, equipping each with a **Role**, **Goal**, and **Backstory** to guide how they behave.
* **Example:** The `Emergency_team` agent is told its role is "Emergency service team" and its backstory is to ensure emergency services are provided based on alerts from other teams.
* **LLM Setup:** She powered all of them using the Gemini model (`llm=gemini_llm`).

### 2. Tools (The Equipment)
By themselves, LLMs can only generate text based on their training data. CrewAI allows you to give agents "Tools" (Python functions) so they can interact with the real world. 
* **Example:** She gave the `Expert_team` agent custom tools like `local_img_interpreter` (to "see" images via Gemini Vision) and `offline_pdf_search_tool` (to "read" the PDF). 
* **Implementation:** She used the `@tool` decorator in her helper scripts (like `global_annotation_tool.py`) to convert standard Python functions into tools the CrewAI agents know how to use.

### 3. Tasks (The Assignments)
A Task is a specific job that needs to be done. It defines what to do, who does it, and what the final deliverable should look like.
* **Example:** She created `task1` through `task6`. For `task1`, she assigned the `Expert_team` agent, provided a detailed `description` of the steps to take, set an `expected_output` format, and told CrewAI to automatically save the final answer to an `output_file` named `Expert_team_report.txt`.
* **Memory:** She set `memory=True` on earlier tasks so that later agents (like the `Alerts_team` in `task2`) could read and build upon the work the `Expert_team` just finished.

### 4. The Crew (The Manager)
The `Crew` binds the Agents and Tasks together and decides how the work is executed. 
* **Implementation:** At the very bottom of the script, she instantiated the `Crew`, handed it the list of `agents` and `tasks`, and set the process to `Process.sequential`. 
* **Execution:** When she calls `crew.kickoff()`, the framework automatically starts `task1`, waits for the `Expert_team` to finish it, passes the output to `task2` for the `Alerts_team`, and so on down the line until all 6 tasks are complete!

---

## 1. `DisasterManagement_teamwork_simulation.py`
**Purpose**: This is the main orchestration script that defines the agents, tools, tasks, and runs the simulation pipeline.

**How it works**:
- **Agents Definition**: It defines multiple CrewAI `Agent` objects, assigning them specific roles, goals, and providing them access to custom and web-search tools.
```python
Expert_team = Agent(
    role="Expert team",
    goal="Analyse the contents got from the detailed observations from the local and global disaster images",
    backstory="""Your primary role is to function as the expert team, generate a disaster report according to the 
    summary of the the content in the images""",
    verbbose=True,
    allow_delegation=True,
    llm=gemini_llm,
    tools=[local_img_interpreter, global_map_annotation, offline_pdf_search_tool] + langchain_tools
)
```

- **Tasks Definition**: The script breaks down the workflow into 6 sequential `Task` objects, linking each to a specific agent and expecting a formatted output written directly to a text file.
```python
task1 = Task(
    description="""Generate descriptions for post-earthquake images using the "local_img_interpreter"
    and the location stickers in the images. Then, get post-earthquake grading with their location name using "offline_pdf_search_tool"
    to make a concise summary (e.g., street A: G1, street B: G4). After that, send 
    the concise summary text to the "global_map_annotation" tool for map annotation. finally generate a new report 
    around 2000 words, with description for the images, region location information, and disaster grade.
    """,
    agent=Expert_team,
    expected_output="...", # (Truncated for brevity)
    memory=True,  ## temporarily save the output
    output_file='Expert_team_report.txt',
)
```

- **Execution**: The agents and tasks are grouped into a CrewAI `Crew` and executed sequentially.
```python
crew = Crew(
    agents=agents,
    tasks=[task1, task2, task3, task4, task5, task6],
    process=Process.sequential,
    verbose=True,
    max_rpm=2 
)

result = crew.kickoff()
```

---

## 2. `global_annotation_tool.py`
**Purpose**: Provides the `global_map_annotation` tool for the Expert agent to visually annotate disaster grades on a map image.

**How it works**:
- The tool uses the `gemini-3.1-flash-lite` vision model to analyze an image, find specific text-based locations, and output their coordinates and disaster grades (e.g., G1-G10) in a strict JSON format.
```python
@tool
def global_map_annotation(text: str) -> str:
    # ...
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
    # ... Uses model.generate_content() to get the JSON layout
```
- Once the JSON is parsed, it uses the Python Imaging Library (`PIL.ImageDraw`) to draw yellow text annotations directly onto the map image and saves it as `4_anotated.png`.

---

## 3. `global_img_tool.py`
**Purpose**: Provides the `global_img_interpreter` tool for agents to analyze global map images and identify dangerous areas.

**How it works**:
- It loads and base64-encodes images from `Disaster_image/Global_image` and passes them to the Gemini vision model along with a prompt.
```python
@tool
def global_img_interpreter(text: str) -> str:
    """Process the text with the LLM."""
    prompt = "Analyze the map for potential dangerous areas. " + text
    
    contents = [prompt]
    for image in encoded_images:
        contents.append({
            "mime_type": "image/png",
            "data": image
        })

    # Implements retry logic for API Rate Limits
    for attempt in range(5):
        try:
            response = model.generate_content(contents)
            return response.text
        # ... catch rate limits and sleep
```

---

## 4. `local_img_tool.py`
**Purpose**: Provides the `local_img_interpreter` tool to analyze ground-level (local) disaster images.

**How it works**:
- Similar to the global image tool, it loads images from the `Disaster_image/Local_image` directory. The primary difference is the prompt it uses to extract detailed ground-level damage assessments.
```python
@tool
def local_img_interpreter(text: str) -> str:
    # ...
    prompt = "describe the disaster in detail in the images, each image requires 200 words. " + text
    # ...
    # Uses model.generate_content() to generate the descriptions
```

---

## 5. `pdf_search_tool.py`
**Purpose**: Provides the `offline_pdf_search_tool` to allow agents to query a local disaster guideline PDF.

**How it works**:
- Implements a basic Retrieval-Augmented Generation (RAG) pipeline using LangChain. It embeds the PDF text chunks into a local `Chroma` vector store using `GoogleGenerativeAIEmbeddings` and retrieves the most relevant chunks via a similarity search.
```python
@tool
def offline_pdf_search_tool(query: str) -> str:
    """Semantic search over a local earthquake guidelines PDF using Gemini Embeddings."""
    # ...
    # Load and split the PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Store in Chroma Vector DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    # Run Similarity Search
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(results)])
```

---

# CrewAI Tools Deep Dive: How They Work Line-by-Line

In CrewAI, LLM agents are inherently just text generators. To allow them to interact with the outside world (like "seeing" images or "reading" a PDF), you must give them **Tools**. 

A CrewAI tool is simply a standard Python function that is wrapped with a special `@tool` decorator. The most critical part of a tool is its **Docstring**—the LLM reads this docstring to understand what the tool does and when it should decide to use it.

Below is a line-by-line breakdown of how Ridita built and integrated her custom tools.

---

## 1. How a Vision Tool is Made: `local_img_tool.py`
This script allows the agent to "see" and describe local disaster photos.

### The Setup and Image Encoding
Before the tool is defined, the script sets up the Gemini Vision model and prepares the images.
```python
# Lines 19-24: Helper function to resize and encode an image into Base64 format. 
# Gemini requires images to be sent as Base64 strings.
def encode_image(image_path, target_size):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
    with io.BytesIO() as buffer:
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Lines 26-33: Helper function that loops through the "Local_image" folder, 
# finds all JPEGs/PNGs, and encodes them into a list of Base64 strings.
def encode_images_in_folder(folder_path,target_size=(256,256)):
    # ... (loops through folder and calls encode_image)
    return encoded_image

# Line 36: Actually executes the function, storing the encoded images in memory.
encoded_images = encode_images_in_folder(folder_path, target_size=(720,720))

# Line 38: Initializes the Gemini Vision model.
model = genai.GenerativeModel("gemini-3.1-flash-lite")
```

### The Tool Definition
This is the actual tool the CrewAI agent will use.
```python
# Line 40: The @tool decorator is what tells CrewAI "this function is a tool an agent can use".
@tool
def local_img_interpreter(text: str) -> str:
    # Lines 42-50: THIS IS CRITICAL. The agent reads this docstring to know how to use the tool. 
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The input from the LLM.
    """
    
    # Line 52: Hardcodes a prompt instruction and appends whatever text the agent sent to the tool.
    prompt = "describe the disaster in detail in the images, each image requires 200 words. " + text
    
    # Lines 55-60: Prepares the payload for Gemini. It adds the text prompt, then loops through
    # all the encoded images and adds them to the payload.
    contents = [prompt]
    for image in encoded_images:
        contents.append({
            "mime_type": "image/png",
            "data": image
        })

    # Lines 62-72: Makes the API call to Gemini. 
    # It uses a `for` loop to retry up to 5 times.
    import time
    for attempt in range(5):
        try:
            # If successful, returns the text description back to the CrewAI Agent.
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            # If a rate limit error (429) occurs, it pauses the whole script for 60 seconds before trying again.
            if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                time.sleep(60) 
            else:
                return f"An error occurred: {str(e)}"
```
*(Note: `global_img_tool.py` works exactly the same way, but points to the `Global_image` folder and uses a different prompt).*

---

## 2. How the RAG Tool is Made: `pdf_search_tool.py`
This tool gives the agent the ability to search a local PDF for earthquake guidelines.

```python
# Line 12: Marks this function as a CrewAI tool.
@tool
def offline_pdf_search_tool(query: str) -> str:
    # Line 14: The docstring telling the agent this is for searching the earthquake guidelines PDF.
    """Semantic search over a local earthquake guidelines PDF using Gemini Embeddings."""

    # Lines 19-22: Sets up Google's embedding model. This turns text into numbers (vectors) 
    # so we can calculate how similar two pieces of text are.
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # Lines 25-29: Checks if the vector database (Chroma) already exists on the hard drive. 
    # If it does, load it.
    if os.path.exists(CHROMA_DB_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_model)
    
    # Lines 30-43: If the database DOES NOT exist, build it from scratch.
    else:
        # 1. Load the PDF file.
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        # 2. Split the massive PDF text into smaller 500-character chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # 3. Embed those chunks into vectors and save them to the Chroma database folder.
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embedding_model, persist_directory=CHROMA_DB_DIR
        )

    # Line 46: Takes the agent's `query`, turns it into a vector, and finds the 3 most similar chunks in the PDF.
    results = vectorstore.similarity_search(query, k=3)

    # Line 49: Joins the 3 chunks together into one big string and returns it to the agent.
    return "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(results)])
```

---

## 3. How the Map Annotation Tool is Made: `global_annotation_tool.py`
This tool does two things: It asks Gemini Vision for coordinates, and then writes text onto an image.

```python
@tool
def global_map_annotation(text: str) -> str:
    """Process the text with the LLM."""
    
    # Lines 77-85: A massive prompt instructing the Gemini model to output a very strict JSON format 
    # containing X/Y coordinates and disaster grades.
    base_instruction = (
        "these images are the same place that experienced disaster. "
        "There are different disaster grade such as G1~G10... "
        "generate a json structure for all the relevant disaster locations with position and grading information "
        'to annotate these labels in the second image like this: '
        '{"annotations": [{"position": [820, 380], "text": "G1"}, {"position": [660, 620], "text": "G2"}]}. '
    )
    prompt = base_instruction + text
    
    # ... (Payload is prepared and Gemini API is called exactly like in local_img_tool.py)
    # response.text now contains a string that looks like: '```json {"annotations": [...]} ```'

    # Lines 116-134: Uses Regex (`re.search`) to strip away the markdown formatting (```json) 
    # and uses `json.loads` to convert the string into a real Python dictionary.
    json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
    # ... (regex cleanup)
    content_dict = json.loads(json_str)

    # Lines 136-139: If the JSON was valid, it calls a helper function `annotate_image`
    if content_dict and 'annotations' in content_dict:
        image_path = os.path.join(script_dir, "Disaster_image", "Global_image", "2.jpg")
        output_path = os.path.join(script_dir, "Disaster_image", "Global_image", "4_anotated.png")
        
        # This helper function (defined at the top of the file) uses the `PIL` library 
        # to draw the yellow text onto the image at the exact X/Y coordinates Gemini provided!
        annotate_image(image_path, content_dict['annotations'], output_path, size=(1080, 1522))
        
    return text
```

---

## 4. Hooking the Tools to the Agents: `DisasterManagement_teamwork_simulation.py`
Finally, how do the agents actually get access to these tools?

```python
# Lines 8-12: The script imports the custom tool functions from the other files.
from local_img_tool import local_img_interpreter
from global_img_tool import global_img_interpreter
from global_annotation_tool import global_map_annotation
from pdf_search_tool import offline_pdf_search_tool

# Lines 29-49: Ridita also creates two simple tools right in the main script for searching the web.
@tool("Internet_search")
def internet_search(query: str) -> str:
    # Uses DuckDuckGo to search the web
    # ...

# Lines 57-66: When creating an Agent, you simply pass the imported tools into the `tools=` array!
Expert_team = Agent(
    role="Expert team",
    # ...
    # Now, whenever the Expert_team needs to solve a problem, it knows it is allowed to call these specific functions!
    tools=[local_img_interpreter, global_map_annotation, offline_pdf_search_tool] + langchain_tools
)
```
