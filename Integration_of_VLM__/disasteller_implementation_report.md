# Project Report: DisasTeller Implementation and Output Analysis

**Subject:** Multi-Agent Large Vision-Language Model (LVLM) Framework for Post-Disaster Response  
**System Name:** DisasTeller  
**Core Model:** Google Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`)  
**Core Framework:** CrewAI & LangChain  

---

## 1. Executive Summary

**DisasTeller** is an agentic, multi-LVLM framework designed to automate critical tasks in post-disaster response. In the immediate aftermath of a disaster (e.g., earthquake, flood, or fire), quick and coordinated decision-making is vital. However, human coordination can suffer from delays and information bottlenecks. 

This implementation demonstrates a complete, automated pipeline that orchestrates four specialized agents to perform:
1. **Damage Assessment** (using image analysis & semantic guidelines search)
2. **Public Alerting** (generating hazards alerts & social media updates)
3. **Resource Deployment & Infrastructure Triage** (shelter allocations & transport recovery)
4. **Strategic Planning** (governmental reconstruction and public community coordination)

The project runs on a sequential workflow powered by **CrewAI**, leveraging **Google Gemini** for multimodal reasoning (interpreting pictures of destruction and map annotations).

---

## 2. System Architecture & Agent Workflow

The system is structured around four specialized agents that collaborate sequentially:

graph TD
    %% Styling
    classDef agent fill:#f9f0ff,stroke:#b19cd9,stroke-width:2px,color:#000
    classDef input fill:#e6f7ff,stroke:#91d5ff,stroke-width:2px,color:#000
    classDef output fill:#f6ffed,stroke:#b7eb8f,stroke-width:2px,color:#000

    subgraph Inputs
        I1(📸 Local Damage Images):::input
        I2(🗺️ Global Map Image):::input
        I3(📄 Guidelines PDF):::input
    end

    subgraph Workflow
        A{🧑‍🔬 Expert Team}:::agent
        B{🚨 Alerts Team}:::agent
        C{🚑 Emergency Service Team}:::agent
        D{📋 Assignment Team}:::agent
    end

    subgraph Outputs
        Outfiles(📑 Structured Output Reports):::output
    end

    %% Data Flow & Dependencies
    I1 & I3 --> A
    I2 --> B
    
    %% Task Execution & Cumulative Context Passing
    A -->|Task 1: Damage Report| B
    A & B -->|Task 2: Alerts| C
    A & B & C -->|Task 3: Shelters & Hospital Plan| D
    D -->|Tasks 4, 5, 6: Deployments & Reports| Outfiles


### 2.1 The Specialized Agents
*   **Expert Team Agent:** Evaluates on-site field images to identify damaged structures, locations, and overlays safety tags. It performs semantic searches over local emergency guidelines.
*   **Alerts Team Agent:** Interprets findings from the Expert Team to evaluate global maps and define hazard zones, producing emergency alerts and public updates.
*   **Emergency Service Team Agent:** Recommends emergency medical centers, temporary field shelters, and prioritizes critical transport pathways.
*   **Assignment Team Agent:** Handles strategic allocation, including paramedic/rescue deployments, government budgeting drafts, and community advisories.

### 2.2 Orchestrated Tasks
The pipeline executes **6 sequential tasks**:
1.  **Task 1 (Damage Assessment):** Visual inspection of local disaster photos and guideline retrieval to grade destruction from G1 (minimal) to G5 (destroyed).
2.  **Task 2 (Public Warning):** Formulating immediate safety alerts and social media templates.
3.  **Task 3 (Emergency Infrastructure):** Staging hospital and shelter allocations.
4.  **Task 4 (Human Resource Allocation):** Outlining personnel deployment plans.
5.  **Task 5 (Community Communication):** Drafting community-wide safety advisories.
6.  **Task 6 (Government Reconstruction Draft):** Creating detailed budgeting estimates and policy-level reconstruction milestones.

---

## 3. Technical Implementation Details

During the deployment of this project, several critical technical issues were resolved to ensure production-grade stability:

### 3.1 Resolving API Rate Limit Errors
*   **The Issue:** Initial executions using standard models like `gemini-3.5-flash` or `gemini-2.5-flash` were blocked by the Google Gemini free-tier daily request limits (**20 requests per day per project**), causing immediate `429 RESOURCE_EXHAUSTED` crashes.
*   **The Workaround:** 
    *   Transitioned the entire framework to **`gemini-3.1-flash-lite`**, which has a significantly higher daily quota allowance and a **15 Requests Per Minute (RPM)** threshold.
    *   Set a global `max_rpm=2` constraint on the CrewAI orchestrator to keep requests comfortably within the RPM limits.
    *   Preserved all multimodal functionality, allowing the agents to successfully read images and generate detailed outputs.

### 3.2 Improving Path and Dotenv Configurations
*   **The Issue:** Running the script from the root workspace directory resulted in the failure to load environment configurations because the `.env` file was nested inside a subdirectory.
*   **The Workaround:** Upgraded the initialization code to dynamically target the active directory relative to the running script:
    ```python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(script_dir, ".env"))
    ```

---

## 4. Generated Outputs & Analysis

All 6 tasks ran to completion successfully and produced structured reports inside the project workspace directory:

### 4.1 [Expert_team_report.txt](file:///d:/DisasTeller/DisasTeller/Expert_team_report.txt)
*   **Purpose:** Detailed damage assessment.
*   **Key Content:**
    *   Grades destruction based on a standardized 5-point scale (G1 to G5).
    *   **Hama Street (Grade: G5):** Total collapse of masonry corners, upper floors, and blocked roads.
    *   **Collapsed Bridge (Grade: G5):** Waterway bridge sheared off at piers; permanently severed connectivity.
    *   **North Asaichi Street (Grade: G3):** Major cracks in classical facades; high secondary hazard potential.
    *   **Staging Area (Grade: G1):** Wajima Drama Memorial Hall found structurally sound.

### 4.2 [Alerts_team_report_for_social_media.txt](file:///d:/DisasTeller/DisasTeller/Alerts_team_report_for_social_media.txt)
*   **Purpose:** Safety alerting.
*   **Key Content:**
    *   Maps out dangerous "Avoid Zones" for citizens.
    *   Generates a Twitter/Facebook emergency template containing safety icons, warning messages, and instructions to stay clear of the Kawaharata River.

### 4.3 [Emergency_team_report.txt](file:///d:/DisasTeller/DisasTeller/Emergency_team_report.txt)
*   **Purpose:** Shelter and medical staging.
*   **Key Content:**
    *   Sets Wajima Drama Memorial Hall as the central triage center (serving 500-800 displaced residents).
    *   Allocates 2 modular tent cities in safe zones.
    *   Prioritizes bridge repairs (deployment of temporary Bailey bridge crossings).

### 4.4 [Emergency_service_human_resource_allocation.txt](file:///d:/DisasTeller/DisasTeller/Emergency_service_human_resource_allocation.txt)
*   **Purpose:** Actionable rescue logistics.
*   **Key Content:**
    *   Splits field workers into three zones (Hama Street, Collapsed Bridge, North Asaichi Street).
    *   Deploys specific quantities of medical personnel (15 doctors, 30 paramedics), heavy engineers (20 workers), and search & rescue canine units.

### 4.5 [Public_community_report.txt](file:///d:/DisasTeller/DisasTeller/Public_community_report.txt)
*   **Purpose:** Community reassurance.
*   **Key Content:**
    *   A citizen-focused advisory emphasizing structural safety rules, contact numbers for rescue hotlines, and guidelines to avoid secondary landslides.

### 4.6 [Reconstruction_plan.txt](file:///d:/DisasTeller/DisasTeller/Reconstruction_plan.txt)
*   **Purpose:** Financial and infrastructure recovery plan for government submission.
*   **Key Content:**
    *   Estimates rebuild costs based on public engineering benchmarks.
    *   Outlines a phased recovery roadmap (Phase 1: Emergency clearing and bridge bypass, Phase 2: Residential rebuilding, Phase 3: Retrofitting heritage buildings).

---

## 5. Conclusion & Faculty Presentation Speaking Points

When presenting this work to your faculty, you can highlight these key points:
1.  **Labor Division:** Explain how the multi-agent paradigm reduces bottlenecks by delegating specialized tasks (assessment, warnings, allocation, planning) to separate LLM personas, mimicking a real emergency command center.
2.  **Vision Integration:** Point out that the framework doesn't just read text; it uses Vision APIs (`gemini-3.1-flash-lite`) to interpret structural failure details directly from field photographs.
3.  **API Engineering:** Discuss the engineering constraints you navigated—such as bypassing Google's strict free-tier limit of 20 daily requests by implementing Lite models combined with global RPM scheduling.
