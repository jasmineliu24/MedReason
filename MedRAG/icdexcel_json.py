import pandas as pd
import json
import argparse
import os

def convert_excel_to_json(excel_path, output_path=None):
    """
    Convert an Excel file with ICD codes to the required JSON format.
    
    Args:
        excel_path: Path to the Excel file
        output_path: Path to save the JSON file (default: same directory as Excel with .json extension)
    
    Returns:
        Path to the saved JSON file
    """
    try:
        # Read the Excel file
        print(f"Reading Excel file from {excel_path}...")
        df = pd.read_excel(excel_path)
        
        # Check if required columns exist
        required_columns = ["procedure code", "short description", "long description"]
        missing_columns = [col for col in required_columns if col.lower() not in [c.lower() for c in df.columns]]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Map Excel columns to JSON fields
        column_mapping = {}
        for col in df.columns:
            if col.lower() == "procedure code":
                column_mapping["id"] = col
            elif col.lower() == "short description":
                column_mapping["short_description"] = col
            elif col.lower() == "long description":
                column_mapping["long_description"] = col
        
        # Convert to JSON structure
        records = []
        for _, row in df.iterrows():
            record = {
                "id": str(row[column_mapping["id"]]),
                "short_description": row[column_mapping["short_description"]],
                "long_description": row[column_mapping["long_description"]]
            }
            records.append(record)
        
        # Determine output path if not provided
        if output_path is None:
            output_path = os.path.splitext(excel_path)[0] + ".json"
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(records)} records to JSON")
        print(f"Saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error converting Excel to JSON: {e}")
        raise

def clean_data(records):
    """
    Clean and standardize the data
    
    Args:
        records: List of record dictionaries
    
    Returns:
        Cleaned list of records
    """
    cleaned_records = []
    
    for record in records:
        # Convert ID to string if it's not already
        if record["id"] is not None:
            # Handle various ID formats
            if isinstance(record["id"], float):
                # Format float to string without decimal if possible
                if record["id"].is_integer():
                    record["id"] = str(int(record["id"]))
                else:
                    record["id"] = str(record["id"])
            else:
                record["id"] = str(record["id"])
                
        # Handle missing descriptions
        if pd.isna(record["short_description"]):
            record["short_description"] = ""
        
        if pd.isna(record["long_description"]):
            record["long_description"] = record["short_description"]
            
        cleaned_records.append(record)
        
    return cleaned_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Excel file with ICD codes to JSON format")
    parser.add_argument("excel_path", help="Path to the Excel file containing ICD codes")
    parser.add_argument("--output", "-o", help="Path to save the JSON file")
    parser.add_argument("--output_dir", "-d", help="Directory to save the JSON file (default: same as Excel)")
    
    args = parser.parse_args()
    
    # Determine output path
    output_path = args.output
    if output_path is None and args.output_dir is not None:
        filename = os.path.basename(args.excel_path)
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output_dir, f"{base_filename}.json")
    
    # Create output directory if it doesn't exist
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert Excel to JSON
    try:
        # Read the Excel file
        print(f"Reading Excel file from {args.excel_path}...")
        df = pd.read_excel(args.excel_path)
        
        # Check if required columns exist (case-insensitive)
        df.columns = [col.lower() for col in df.columns]
        required_columns = ["procedure code", "short description", "long description"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Rename columns for consistency
        df.rename(columns={
            "procedure code": "id",
            "short description": "short_description",
            "long description": "long_description"
        }, inplace=True)
        
        # Convert to records
        records = df.to_dict('records')
        
        # Clean data
        records = clean_data(records)
        
        # Determine output path if not provided
        if output_path is None:
            output_path = os.path.splitext(args.excel_path)[0] + ".json"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(records)} records to JSON")
        print(f"Saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)