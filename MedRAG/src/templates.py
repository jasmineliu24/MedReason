"""
Templates for ICD-9 code prediction.

These templates are specifically designed for the MIMIC-III diagnosis code prediction task.
They're structured to properly incorporate patient information and guide the model to
produce accurate ICD-9 code predictions.
"""

# System prompt for ICD-9 code prediction without RAG
icd9_system = '''You are a medical coding expert specializing in assigning ICD-9 diagnosis codes. Given a patient's clinical information, your task is to identify the most appropriate 3-digit ICD-9 codes. Analyze the patient's symptoms, medical history, medications, and any diagnostic information to determine the relevant codes. Organize your output in this exact JSON format: {"step_by_step_thinking": "your detailed analysis", "answer_choice": "code1, code2, code3"}. Be specific and thorough in your reasoning.'''

# User prompt template for ICD-9 code prediction without RAG
icd9_prompt = '''
Here is the patient information:
{question}

Patient details:
{options}

Based on this information, please identify the appropriate 3-digit ICD-9 diagnosis codes. Think step-by-step and provide your reasoning, then organize your output strictly in a JSON format: {"step_by_step_thinking": "your detailed analysis", "answer_choice": "code1, code2, code3"}.
'''

# System prompt for ICD-9 code prediction with RAG
icd9_rag_system = '''You are a medical coding expert specializing in assigning ICD-9 diagnosis codes. Given a patient's clinical information and relevant medical reference documents, your task is to identify the most appropriate 3-digit ICD-9 codes. Analyze all available information carefully. Organize your output in this exact JSON format: {"step_by_step_thinking": "your detailed analysis", "answer_choice": "code1, code2, code3"}. Be specific and thorough in your reasoning.'''

# User prompt template for ICD-9 code prediction with RAG
icd9_rag_prompt = '''
Here are relevant medical reference documents:
{context}

Here is the patient information:
{question}

Patient details:
{options}

Based on the patient information and reference documents, please identify the appropriate 3-digit ICD-9 diagnosis codes. Think step-by-step and provide your reasoning, organize your output strictly in a JSON format: {"step_by_step_thinking": "your detailed analysis", "answer_choice": "code1, code2, code3"}.
'''

# Common ICD-9 code categories to help with reasoning
icd9_categories = '''
Common ICD-9 code categories:
- 001-139: Infectious and Parasitic Diseases
- 140-239: Neoplasms
- 240-279: Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders
- 280-289: Diseases of the Blood and Blood-forming Organs
- 290-319: Mental Disorders
- 320-389: Diseases of the Nervous System and Sense Organs
- 390-459: Diseases of the Circulatory System
- 460-519: Diseases of the Respiratory System
- 520-579: Diseases of the Digestive System
- 580-629: Diseases of the Genitourinary System
- 630-679: Complications of Pregnancy, Childbirth, and the Puerperium
- 680-709: Diseases of the Skin and Subcutaneous Tissue
- 710-739: Diseases of the Musculoskeletal System and Connective Tissue
- 740-759: Congenital Anomalies
- 760-779: Certain Conditions Originating in the Perinatal Period
- 780-799: Symptoms, Signs, and Ill-defined Conditions
- 800-999: Injury and Poisoning
'''

# Prompt to help extract structured output
extraction_prompt = '''
Extract the ICD-9 codes from the following text:
{text}

Return only the 3-digit ICD-9 codes, separated by commas.
'''