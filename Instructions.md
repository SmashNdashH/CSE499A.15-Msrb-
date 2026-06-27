```markdown
# Setup & Execution Instructions

### 1. Create a Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate the Virtual Environment
```powershell
.\venv\Scripts\activate
```

### 3. Install Project Dependencies
```powershell
pip install -r requirements.txt
```
*(Note: `generate_ground_truth.py` is self-bootstrapping and will automatically install its own missing dependencies, but this step ensures your entire workspace is ready).*

### 4. Set Your Gemini API Key
**If using PowerShell:**
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```
**If using Command Prompt (CMD):**
```cmd
set GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Run the Script
```powershell
python support\generate_ground_truth.py
```
```