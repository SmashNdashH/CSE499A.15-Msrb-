import json
import os

notebook_path = r'c:\Users\taman\Desktop\CSE499AB\CSE499A.15-Msrb--1\support\Tamanna\DisasterVQA\disastervqa.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Fix 1 & 2: OOM & Flash Attention & model import in the inference cell
        if 'Qwen2VLForConditionalGeneration' in source or 'Qwen2_5_VLForConditionalGeneration' in source:
            source = source.replace('Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration')
            source = source.replace('Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct')
            
            # Avoid Flash Attention Error -> use SDPA instead, and set device_map=auto to prevent OOM
            if 'attn_implementation' not in source:
                source = source.replace('device_map="auto"', 'device_map="auto", attn_implementation="sdpa"')
            
            # Add tqdm import if not present
            if 'from tqdm import tqdm' not in source:
                source = 'from tqdm import tqdm\n' + source
                
            # Fix 3: formatted_samples not defined -> in the notebook it is loaded as `lines`
            # Let's change `lines[:200]` to `tqdm(lines[:200])`
            if 'for sample in lines[:200]:' in source:
                source = source.replace('for sample in lines[:200]:', 'for sample in tqdm(lines[:200]):')
            if 'for sample in tqdm(formatted_samples[:100]):' in source:
                source = source.replace('for sample in tqdm(formatted_samples[:100]):', 'for sample in tqdm(lines[:100]):')
            if 'for sample in tqdm(formatted_samples):' in source:
                source = source.replace('for sample in tqdm(formatted_samples):', 'for sample in tqdm(lines):')
            
        # Fix 4: tqdm not defined in evaluate cell
        if 'Evaluating answers' in source and 'for pred in predictions:' in source:
            if 'from tqdm import tqdm' not in source:
                source = 'from tqdm import tqdm\n' + source
            source = source.replace('for pred in predictions:', 'for pred in tqdm(predictions):')
            
        # Also fix evaluate_open_ended cell if they have another one
        if 'Evaluating answers' in source and 'for pred in tqdm(results):' in source:
            if 'from tqdm import tqdm' not in source:
                source = 'from tqdm import tqdm\n' + source

        cell['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 and not line.endswith('\n') else '') for i, line in enumerate(source.split('\n'))]
        if cell['source'] and cell['source'][-1].endswith('\n') and source and not source.endswith('\n'):
            cell['source'][-1] = cell['source'][-1][:-1]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated notebook successfully!')
