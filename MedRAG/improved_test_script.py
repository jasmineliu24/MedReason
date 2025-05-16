from src.improved_medrag import MedRAG
import json
import argparse
import os
import re
import sys
from tqdm import tqdm

def batch_data(data_list, batch_size=512):
    """Split data into batches"""
    batch_data = []
    for i in range(0, len(data_list), batch_size):
        batch_data.append(data_list[i:i + batch_size])
    return batch_data

def evaluate_icd9_accuracy(predictions, ground_truth):
    """
    Evaluate ICD-9 code prediction accuracy
    
    Args:
        predictions: List of predicted ICD-9 codes
        ground_truth: List of ground truth ICD-9 codes
        
    Returns:
        precision, recall, f1
    """
    total_correct = 0
    total_predicted = 0
    total_ground_truth = 0
    
    for pred, gt in zip(predictions, ground_truth):
        # Extract ICD-9 codes from predictions and ground truth
        if isinstance(pred, str):
            pred_codes = re.findall(r'\b\d{3}\b', pred)  # Extract 3-digit ICD-9 codes
        else:
            pred_codes = []
            
        if isinstance(gt, str):
            gt_codes = re.findall(r'\b\d{3}\b', gt)
        else:
            gt_codes = []
            
        # Convert to sets for easier comparison
        pred_set = set(pred_codes)
        gt_set = set(gt_codes)
        
        # Calculate metrics
        correct = len(pred_set.intersection(gt_set))
        total_correct += correct
        total_predicted += len(pred_set)
        total_ground_truth += len(gt_set)
    
    # Calculate precision, recall, and F1
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def extract_icd9_codes(model_output):
    """Extract ICD-9 codes from model output"""
    # First try to extract from JSON format
    try:
        if '"answer_choice":' in model_output:
            match = re.search(r'"answer_choice":\s*"([^"]+)"', model_output)
            if match:
                answer_text = match.group(1)
                return answer_text
    except:
        pass
    
    # Try other patterns
    patterns = [
        r'ICD-9 Code:\s*([\d, ]+)',  # Matches "ICD-9 Code: 123, 456"
        r'ICD-9 Codes?:\s*([\d, ]+)',  # Matches "ICD-9 Codes: 123, 456"
        r'answer_choice":\s*"([^"]+)"',  # Matches json format
        r'\b(\d{3})\b'  # Matches any 3-digit code
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, model_output)
        if matches:
            return matches.group(1)
    
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--HNSW", action="store_true")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--corpus_name", type=str, default="Textbooks")
    parser.add_argument("--retriever_name", type=str, default="MedCPT")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--dataset_name", type=str, default="mimic-iii")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/clinical-llm-benchmark/dataset_raw/106.MIMIC-III Outcome.Diagnosis.SFT.json")
    parser.add_argument("--icd_codes_path", type=str, default="./corpus/icd_codes.json", 
                      help="Path to JSON file containing ICD-9 and ICD-10 codes")

    args = parser.parse_args()

    # Set up save directory
    if args.rag:
        save_dir = f"{args.results_dir}/{args.dataset_name}/rag_{args.k}/{args.llm_name.split('/')[-1]}/{args.corpus_name}/{args.retriever_name}"
    else:
        save_dir = f"{args.results_dir}/{args.dataset_name}/cot/{args.llm_name.split('/')[-1]}"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving to {save_dir}")

    # Save arguments
    args_dict = vars(args)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # Check if ICD codes file exists
    if not os.path.exists(args.icd_codes_path):
        print(f"Warning: ICD codes file not found at {args.icd_codes_path}")
    else:
        print(f"Found ICD codes file at {args.icd_codes_path}")

    # Initialize MedRAG
    medrag = MedRAG(
        rag=args.rag,
        retriever_name=args.retriever_name,
        corpus_name=args.corpus_name,
        llm_name=args.llm_name,
        HNSW=args.HNSW,
        db_dir="./corpus",
        icd_codes_path=args.icd_codes_path
    )

    # Load dataset
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Extract questions and options
    questions = []
    options = []
    ground_truth = []

    for data in dataset:
        questions.append(data['instruction'])
        options.append(data['input'])
        ground_truth.append(data['output'])

    # Batch the data
    batched_questions = batch_data(questions, args.batch_size)
    batched_options = batch_data(options, args.batch_size)
    batched_ground_truth = batch_data(ground_truth, args.batch_size)

    # Process each batch
    save_dict = []
    all_predictions = []
    all_ground_truth = []

    for batch_idx, (batch_questions, batch_options, batch_gt) in enumerate(
        zip(batched_questions, batched_options, batched_ground_truth)):
        
        print(f"Processing batch {batch_idx+1}/{len(batched_questions)}")
        
        # Get answers from MedRAG
        batch_answers, snippets, scores = medrag.batch_answer(
            questions=batch_questions,
            options=batch_options,
            k=args.k
        )
        
        # Process each answer in the batch
        for i, (question, option, answer, gt) in enumerate(
            zip(batch_questions, batch_options, batch_answers, batch_gt)):
            
            # Extract ICD-9 codes from the model's answer
            extracted_codes = extract_icd9_codes(answer)
            
            # Save results
            save_dict.append({
                "question": question,
                "option": option,
                "generate_text": answer,
                "answer": gt,
                "extracted_codes": extracted_codes
            })
            
            all_predictions.append(extracted_codes)
            all_ground_truth.append(gt)
            
            # Print progress
            print(f"Processed {batch_idx * args.batch_size + i + 1}/{len(questions)}")
            
        # Save intermediate results
        with open(os.path.join(save_dir, "model_output.json"), 'w') as file:
            json.dump(save_dict, file, indent=4)

    # Calculate metrics
    precision, recall, f1 = evaluate_icd9_accuracy(all_predictions, all_ground_truth)
    
    # Print results
    print(f"Total examples: {len(questions)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save metrics
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_examples": len(questions)
    }
    
    with open(os.path.join(save_dir, "metrics.json"), 'w') as file:
        json.dump(metrics, file, indent=4)
        
    with open(os.path.join(save_dir, "result.txt"), 'w') as file:
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")