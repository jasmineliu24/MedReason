import json
import re
from collections import defaultdict

def extract_codes_from_generate_text(text):
    """
    Extract diagnosis codes from generate_text field, handling both ICD-9 and ICD-10 formats
    """
    extracted_codes = []
    
    # First try to find answer_choice in JSON format (ICD-10 codes)
    answer_choice_patterns = [
        r'"answer_choice":\s*"([^"]+)"',  # Match "answer_choice": "codes"
        r'"answer_choice":\s*\[([^\]]+)\]'  # Match "answer_choice": [codes]
    ]
    
    for pattern in answer_choice_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Find all codes in the match (both ICD-9 and ICD-10)
            codes = re.findall(r'\b([A-Za-z]\d{2,3}(?:\.\d+)?|\d{3,4}(?:\.\d+)?)\b', match)
            extracted_codes.extend(codes)
    
    # If still no codes, look for explicit code listing
    if not extracted_codes:
        code_list_pattern = r'(?:ICD[- ]?(?:9|10)[^:]*:\s*|Answer:\s*)([A-Za-z0-9\., ]+)'
        match = re.search(code_list_pattern, text, re.IGNORECASE)
        if match:
            codes_str = match.group(1)
            codes = re.findall(r'\b([A-Za-z]\d{2,3}(?:\.\d+)?|\d{3,4}(?:\.\d+)?)\b', codes_str)
            extracted_codes.extend(codes)
    
    # If still no codes, look for any potential codes in the text
    if not extracted_codes:
        potential_codes = re.findall(r'\b([A-Za-z]\d{2,3}(?:\.\d+)?|\d{3,4}(?:\.\d+)?)\b', text)
        # Filter out unlikely codes
        filtered_codes = []
        for code in potential_codes:
            # Must be at least 3 characters, not a year, etc.
            if (len(code) >= 3 and not re.match(r'^(19|20)\d{2}$', code)):
                filtered_codes.append(code)
        extracted_codes.extend(filtered_codes)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []
    for code in extracted_codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    return unique_codes

def clean_codes(codes):
    """
    Clean the extracted codes by removing decimal points
    Handles both ICD-9 and ICD-10 formats
    """
    cleaned = []
    for code in codes:
        # Remove decimal point if present
        clean_code = code.replace('.', '')
        # For ICD-9 codes, take first 3 digits
        if clean_code.isdigit():
            clean_code = clean_code[:3]
        cleaned.append(clean_code)
    return cleaned

def clean_codes(codes):
    """
    Clean the extracted codes by removing decimal points and everything after them.
    
    Args:
        codes (list): List of extracted ICD-9 codes
    
    Returns:
        list: List of cleaned ICD-9 codes
    """
    cleaned = []
    for code in codes:
        # Remove decimal point and everything after it
        clean_code = code.split('.')[0]
        if clean_code:  # Ensure we don't add empty strings
            cleaned.append(clean_code)
    return cleaned

def extract_ground_truth_codes(answer_text):
    """
    Extract ground truth codes from the answer field.
    
    Args:
        answer_text (str): The answer text
    
    Returns:
        list: List of ground truth ICD-9 codes
    """
    ground_truth_codes = []
    
    # Look for "ICD-9 Code: code1, code2, ..."
    if 'ICD-9 Code:' in answer_text:
        codes_section = answer_text.split('ICD-9 Code:')[1].strip()
        
        # Handle comma-separated values
        if ',' in codes_section:
            ground_truth_codes = [code.strip() for code in codes_section.split(',')]
        # Handle space-separated values
        else:
            ground_truth_codes = codes_section.split()
    
    return ground_truth_codes

def calculate_metrics(data):
    """
    Calculate accuracy metrics by comparing predicted codes to ground truth.
    
    Args:
        data (list): The processed data
        
    Returns:
        dict: Dictionary of metrics
    """
    total_records = len(data)
    records_with_predictions = 0
    total_ground_truth_codes = 0
    total_predicted_codes = 0
    total_correct_predictions = 0
    
    # For each record
    for record in data:
        ground_truth = record.get('cleaned_ground_truth', [])
        predictions = record.get('cleaned_codes', [])
        
        if predictions:
            records_with_predictions += 1
        
        total_ground_truth_codes += len(ground_truth)
        total_predicted_codes += len(predictions)
        
        # Count correct predictions (intersection)
        correct = len(set(ground_truth) & set(predictions))
        total_correct_predictions += correct
    
    # Calculate metrics
    metrics = {
        "total_records": total_records,
        "records_with_predictions": records_with_predictions,
        "records_with_predictions_percentage": round(records_with_predictions / total_records * 100, 2) if total_records > 0 else 0,
        "total_ground_truth_codes": total_ground_truth_codes,
        "total_predicted_codes": total_predicted_codes,
        "total_correct_predictions": total_correct_predictions,
        "precision": round(total_correct_predictions / total_predicted_codes * 100, 2) if total_predicted_codes > 0 else 0,
        "recall": round(total_correct_predictions / total_ground_truth_codes * 100, 2) if total_ground_truth_codes > 0 else 0
    }
    
    # Calculate F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = round(2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]), 2)
    else:
        metrics["f1_score"] = 0
    
    return metrics

def process_file(file_path):
    """
    Process the input file and update it with extracted and cleaned codes.
    Save results to a new file with "cleaned_" prefix.
    
    Args:
        file_path (str): Path to the input file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # If the file is not valid JSON, try to parse it as a JSONL file
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line: {line}")

    # If data is a dictionary, convert it to a list
    if isinstance(data, dict):
        data = [data]

    # Process each record
    records_processed = 0
    records_with_extracted_codes = 0
    
    for record in data:
        records_processed += 1
        
        # Extract and clean codes ONLY from generate_text
        if 'generate_text' in record:
            extracted_codes = extract_codes_from_generate_text(record['generate_text'])
            record['extracted_codes'] = extracted_codes
            record['cleaned_codes'] = clean_codes(extracted_codes)
            
            if extracted_codes:
                records_with_extracted_codes += 1
        else:
            print(f"Warning: No generate_text field found in record {records_processed}")
            record['extracted_codes'] = []
            record['cleaned_codes'] = []
        
        # Extract ground truth codes from answer field
        if 'answer' in record:
            ground_truth_codes = extract_ground_truth_codes(record['answer'])
            record['ground_truth_codes'] = ground_truth_codes
            record['cleaned_ground_truth'] = clean_codes(ground_truth_codes)
    
    # Calculate accuracy metrics
    metrics = calculate_metrics(data)
    
    # Add metrics to the data
    data_with_metrics = {
        "data": data,
        "metrics": metrics
    }
    
    # Print statistics
    print(f"Processed {records_processed} records")
    print(f"Successfully extracted codes for {records_with_extracted_codes} records")
    print(f"Precision: {metrics['precision']}%")
    print(f"Recall: {metrics['recall']}%")
    print(f"F1 Score: {metrics['f1_score']}")

    # Create output filename with "cleaned_" prefix
    import os
    input_filename = os.path.basename(file_path)
    output_filename = f"cleaned_{input_filename}"
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    
    # Write the updated data to the new file
    try:
        with open(output_path, 'w') as f:
            json.dump(data_with_metrics, f, indent=4)
        print(f"Successfully processed {file_path}")
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")

def main():
    """
    Main function to handle command line arguments and process files.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python icd9_extractor.py input_file.json")
        return
    
    file_path = sys.argv[1]
    process_file(file_path)

if __name__ == "__main__":
    main()