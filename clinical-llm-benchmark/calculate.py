import json
import re

def main():
    # Load the dataset
    file_path = "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/clinical-llm-benchmark/result/106.MIMIC-III Outcome.Diagnosis/deepseek8b/106.MIMIC-III Outcome.Diagnosis-direct-greedy-42.result.json"
    with open(file_path, "r") as file:
        data = json.load(file)
    print("Dataset loaded.")

    def extract_pred_codes(pred_text):
        """
        Extracts the ICD-9 procedure codes from the 'pred' field.
        The codes are expected to appear after the phrase 'ICD-9 procedure codes:'.
        """
        match = re.search(r"ICD-9 Diagnosis codes:\s*(.+)", pred_text, re.IGNORECASE)
        if match:
            codes_str = match.group(1)
            codes = [re.sub(r'\..*', '', code.strip()) for code in codes_str.split(',')]
            return sorted(set(codes))
        return []

    def extract_output_codes(output_text):
        match = re.search(r"ICD-9 Code:\s*(.+)", output_text)
        if match:
            codes = [re.sub(r'\..*', '', code.strip()) for code in match.group(1).split(',')]
            return sorted(set(codes))
        return []

    # Process each item in the dataset
    results = []
    exact_match = half_or_more_match = any_match = 0

    for item in data:
        true_codes = extract_output_codes(item['output'])
        pred_codes = extract_pred_codes(item['pred'])

        true_set, pred_set = set(true_codes), set(pred_codes)
        all_match = true_set == pred_set
        intersection_len = len(true_set & pred_set)

        results.append({
            "id": item["id"],
            "true_codes": true_codes,
            "pred_codes": pred_codes,
            "all_match": all_match,
            "half_match": intersection_len >= len(true_set) / 2,
            "some_match": intersection_len > 0
        })

        exact_match += all_match
        half_or_more_match += intersection_len >= len(true_set) / 2
        any_match += intersection_len > 0

    # Accuracy metrics
    total = len(data)
    accuracy_metrics = {
        "exact_match_accuracy": exact_match / total,
        "half_or_more_match_accuracy": half_or_more_match / total,
        "any_match_accuracy": any_match / total
    }

    # Save results
    output_file_path = "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/clinical-llm-benchmark/result/processed/diagnosis/deepseek8b.json"
    with open(output_file_path, "w") as out_file:
        json.dump({"results": results, "accuracy": accuracy_metrics}, out_file, indent=2)

    print("Results saved to:", output_file_path)
    print("Accuracy metrics:", accuracy_metrics)

if __name__ == "__main__":
    main()
