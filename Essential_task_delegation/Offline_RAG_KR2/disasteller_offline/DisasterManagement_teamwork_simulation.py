"""
Copy of support/Ridita_Integration_of_VLM__/DisasTeller/DisasterManagement_teamwork_simulation.py,
offline by default (System_Design_Notes §2). The original is unchanged.

Changes from the original:
  - The two web tools and the "search the internet" wording of tasks 3-6 are used only
    with DISASTELLER_ONLINE=1. Offline, tasks 3-6 tell the agents to use the guideline
    search and the facts they are given, and to write "not available offline" for
    anything else, such as costs.
  - The Emergency service and Assignment teams also get the guideline search.
  - Reports go to reports/offline/ or reports/online/, and each starts with its mode.
  - CrewAI's usage telemetry is off.
The LLM and the three image tools still call Gemini (roadmap T2 steps 1-3).
"""

import os

# CrewAI otherwise sends usage telemetry to telemetry.crewai.com
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

from crewai import Agent, Task, Crew, Process, LLM

from langchain.tools import tool as langchain_tool
from crewai.tools import tool
from local_img_tool import local_img_interpreter
from global_img_tool import global_img_interpreter
from global_annotation_tool import global_map_annotation
from pdf_search_tool import offline_pdf_search_tool

from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))

# Set up Gemini LLM
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

gemini_llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    api_key=api_key
)

# Offline by default. DISASTELLER_ONLINE=1 adds the two web tools and the
# "search the internet" wording of tasks 3-6.
ONLINE = os.getenv("DISASTELLER_ONLINE") == "1"
MODE = "online" if ONLINE else "offline"
print(f"DisasTeller mode: {MODE}")

if ONLINE:
    import requests
    from bs4 import BeautifulSoup
    from duckduckgo_search import DDGS

    # Define tools for the agent
    @tool("Internet_search")
    def internet_search(query: str) -> str:
        """Useful for querying content on the internet using DuckDuckGo """
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=2)]
            print("use DDGS searching")
            return str(results) if results else "No Results"

    @tool("Process_search_results")
    def process_search_results(url: str) -> str:
        """Process content from webpages"""
        response = requests.get(url=url)
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text()
        truncated_content = text_content[:10000] + "..." if len(text_content) > 10000 else text_content
        return truncated_content

    langchain_tools = [
        internet_search,
        process_search_results,
    ]
else:
    langchain_tools = []

Expert_team_tools = [
    local_img_interpreter,
    global_map_annotation,
    offline_pdf_search_tool,
] + langchain_tools

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

Alerts_team = Agent(
    role="Alerts team",
    goal="Analyse the contents got from the 'Expert team', social media news writing and danger area alerting",
    backstory="""Your primary role is to function as the Alerts team, generate danger area alert and social media news
     according to the contents got from the 'Expert team'. """,
    verbbose=True,
    allow_delegation=True,
    llm=gemini_llm,
    tools=[global_img_interpreter] + langchain_tools
)

Assign_team = Agent(
    role="Assignment team",
    goal="Analyse the contents got from other teams, assign human resource",
    backstory="""Your primary role is to function as the Assignment team, allocate human resource for carrying out the 
    post-disaster management, and finally inform the public communities about your organisation's operation, and 
    generate the reconstruction plan. """,
    verbbose=True,
    allow_delegation=False,  # forbid it reuse the global_map_annotation in 'expert team'
    llm=gemini_llm,
    tools=[global_img_interpreter, offline_pdf_search_tool] + langchain_tools
)

Emergency_team = Agent(
    role="Emergency service team",
    goal="Analyse the contents got from 'Expert team' and 'Alerts team', ensure emergency services providing",
    backstory="""Your primary role is to function as the Emergency service team, according to area alerts from 
    'Alerts team', and the local inspection from 'Expert team', to ensure the locations and quantity of emergent 
    shelter and hospital, confirm the location of immediate transportation recovery.""",
    verbbose=True,
    allow_delegation=False,  # forbid it to use other agents
    llm=gemini_llm,
    tools=[local_img_interpreter, global_img_interpreter, offline_pdf_search_tool] + langchain_tools
)

agents = [Expert_team, Alerts_team, Emergency_team, Assign_team]  # Add other agents as needed

# Offline, tasks 3-6 get this instead of "search the internet" (System_Design_Notes §2.2)
OFFLINE_RULES = """
    Work offline: use only the facts given to you (the other teams' reports and the image tools) and the
    guideline search "offline_pdf_search_tool", and cite the section IDs it returns, e.g. [SPHERE-3-p1].
    If something needs data you do not have, such as costs, budgets, staff numbers from past disasters,
    population counts or contact details, write "not available offline" instead of estimating it.
    """

task1 = Task(
    description="""Generate descriptions for post-earthquake images using the "local_img_interpreter"
    and the location stickers in the images. Then, get post-earthquake grading with their location name using "offline_pdf_search_tool"
    to make a concise summary (e.g., street A: G1, street B: G4). After that, send 
    the concise summary text to the "global_map_annotation" tool for map annotation. finally generate a new report 
    around 2000 words, with description for the images, region location information, and disaster grade.
    """,
    agent=Expert_team,
    expected_output=""" Your final response should generate a comprehensive report that:
    1. Describes the images and location name in lots of detail using the "local_img_interpreter".
    2. Analyzes the images, image stickers, and PDF to determine the disaster grade of different locations using the 
     "offline_pdf_search_tool". Get a concise summary of the disaster region location name with their relevant grading.
    3. send the concise summary text to the "global_map_annotation" tool for map annotation.
    4. Generate a report that should be around 2000 words with additional description from "local_img_interpreter" and 
    formatted clearly for presentation.
    """,
    memory=True,  ## temporarily save the output
    output_file=f'reports/{MODE}/Expert_team_report.txt',
)

task2 = Task(
    description=""" Analyse the output from the "task1" of Expert team, generate danger area alerting and social media news for
    the disaster.
    """,
    agent=Alerts_team,
    expected_output=""" Your final response should generate a comprehensive social media news that:
    1. Describes the damaged events for regional areas, with location information and disaster grade, analyzes potential
     dangerous area according to the global map using "global_img_interpreter".
    """,
    memory=True, ## temporarily save the output
    output_file=f'reports/{MODE}/Alerts_team_report_for_social_media.txt',
)

task3 = Task(
    description=""" analyse the output from the "task1" of Expert team and "task2" of Alerts team. Then, ensure the locations and detailed number of 
    emergent shelter and hospital using "global_img_interpreter" and searching the internet to take a reference from other area historical disaster data. Also, confirm which place needs 
    immediate transportation recovery.
    """ if ONLINE else (""" analyse the output from the "task1" of Expert team and "task2" of Alerts team. Then, ensure the locations and detailed number of
    emergent shelter and hospital using "global_img_interpreter" and the guideline search "offline_pdf_search_tool" (Sphere for
    shelter: living space and site area per person, location and access; INSARAG for handing rescued people over to hospitals). Also, confirm which place needs
    immediate transportation recovery.
    """ + OFFLINE_RULES),
    agent=Emergency_team,
    expected_output=""" Your final response should generate a comprehensive report that contains the information of 
    locations and detailed number (such as 3) for the emergent shelter and hospital, and confirm which place needs 
    immediate transportation recovery.
    """,
    memory=True, ## temporarily save the output
    output_file=f'reports/{MODE}/Emergency_team_report.txt',
)

task4 = Task(
    description=""" Analyse the output from the "task1", "task2", and "task3", Generate a report around 2000 words for 
    the emergency service team human resource allocation, search the internet to take a reference from other area historical disaster data.
    """ if ONLINE else (""" Analyse the output from the "task1", "task2", and "task3", Generate a report around 2000 words for
    the emergency service team human resource allocation. Use the guideline search "offline_pdf_search_tool" for INSARAG
    (rescue team types, ASR levels, worksite triage and priorities, shifts).
    """ + OFFLINE_RULES),
    agent=Assign_team,
    expected_output=""" 
       Emergency Service Team Human Resource Allocation Report:
    - **Introduction**: Brief overview of the disaster and its impact.
    - **Current Situation**: Summary of the damage and immediate needs.
    - **Human Resource Requirements**: Detailed breakdown of human resources required in different areas, Categories of resources needed (e.g., the number of medical personnel, the number of rescue teams, logistics support).
    - **Allocation Plan**: Allocation of human resources to various affected areas, including:
        - **Area A**: Number and type of personnel required, current deployment status.
        - **Area B**: Number and type of personnel required, current deployment status.
        - **Area C**: Number and type of personnel required, current deployment status.
        - (Continue as necessary for other affected areas.)
    - **Challenges and Considerations**: Potential challenges in resource allocation (e.g., transportation issues, local restrictions). Considerations for adjusting allocation based on evolving needs and feedback from the field.
    - **Recommendations**: Suggested adjustments to current allocation based on observed needs and gaps. Recommendations for additional resources or support if necessary.
    - **Conclusion**: Summary of key findings and next steps, with emphasis on the ongoing efforts and future plans.

    """,
    output_file=f'reports/{MODE}/Emergency_service_human_resource_allocation.txt',
)

task5 = Task(
    description=""" Analyse the output from the "task1", "task2", and "task3", search the internet and 
    Generate a report around 2000 words to inform the public communities.
    """ if ONLINE else (""" Analyse the output from the "task1", "task2", and "task3", and
    Generate a report around 2000 words to inform the public communities. Use the guideline search "offline_pdf_search_tool"
    for shelter and household items (Sphere) and rescue operations (INSARAG).
    """ + OFFLINE_RULES),
    agent=Assign_team,
    expected_output=""" 
       Public Community Report:
    - **Introduction**: Summary of the disaster, affected areas, and initial observations from the output of "task1" (Expert team) and "task2" (Alerts team).
    - **Current Situation**: Detailed description of the damage and immediate emergency services being provided by the output of "task3" (Emergency service team).
    - **Resource Allocation**: Information on human resources and logistics for the emergency services, including how resources are being allocated to different areas based on current needs.
    - **Public Advisory**: Instructions and advice for the public on how to stay safe, access emergency services, and cooperate with ongoing operations.
    - **Organizational Operation**: Overview of your organization’s role, operations, and contact information for further assistance.
    - **Conclusion**: Encouraging words and reassurances to the community.
    """,
    output_file=f'reports/{MODE}/Public_community_report.txt',
)

task6 = Task(
    description=""" Analyse the output from the "task1", "task2", and "task3", search the internet and 
    Generate a reconstruction plan report with financial budgets around 4000 words with no space, aiming to submit it 
    to the government and policy maker.
    """ if ONLINE else (""" Analyse the output from the "task1", "task2", and "task3", and
    Generate a reconstruction plan report around 4000 words with no space, aiming to submit it
    to the government and policy maker. Use the guideline search "offline_pdf_search_tool" for Sphere (repair, retrofit
    or rebuild, debris, building codes) and EMS-98 (damage grades). Every cost and budget figure is "not available offline".
    """ + OFFLINE_RULES),
    agent=Assign_team,
    expected_output=""" 
       Reconstruction Plan Report:
    - **Executive Summary**: Overview of the disaster impact and immediate response efforts.
    - **Assessment of Damage**: Detailed analysis of the damage, including data from the 'Expert team' and assessments of affected areas.
    - **Reconstruction Needs**: Searching the internet to take a reference from other area historical disaster data, analyse especially how many money needed, and the required human resources and materials for reconstruction in different areas.
    - **Plan of Action**: Searching the internet to take a reference from other area historical disaster data, detailed plan outlining the steps for reconstruction, including timelines, resource allocation, and coordination with local authorities.
    - **Recommendations**: Suggestions for policy and funding to support effective and efficient reconstruction.
    - **Appendices**: Supporting data, maps, and any additional relevant information.
    """ if ONLINE else """
       Reconstruction Plan Report:
    - **Executive Summary**: Overview of the disaster impact and immediate response efforts.
    - **Assessment of Damage**: Detailed analysis of the damage, including data from the 'Expert team' and assessments of affected areas.
    - **Reconstruction Needs**: The required human resources and materials for reconstruction in different areas, from the damage facts and the guidelines (repair, retrofit or rebuild). Money needed: "not available offline".
    - **Plan of Action**: Detailed plan outlining the steps for reconstruction, including their order, resource allocation, and coordination with local authorities, following the guidelines.
    - **Recommendations**: Suggestions for policy and funding to support effective and efficient reconstruction.
    - **Appendices**: Supporting data, maps, and any additional relevant information.
    """,
    output_file=f'reports/{MODE}/Reconstruction_plan.txt',
)

crew = Crew(
    agents=agents,
    tasks=[task1, task2, task3, task4, task5, task6],
    process=Process.sequential,
    verbose=True,
    max_rpm=2 
)

result = crew.kickoff()

# Each report states the mode that produced it (System_Design_Notes §2.2)
for task in crew.tasks:
    with open(task.output_file, encoding="utf-8") as f:
        report = f.read()
    with open(task.output_file, "w", encoding="utf-8") as f:
        f.write(f"Mode: {MODE}\n\n{report}")















