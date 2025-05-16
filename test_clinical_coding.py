import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from vllm import LLM, SamplingParams

# Set model paths here
MODEL_PATHS = {
    "llama-3.1-8b": "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/Llama-3.1-8B-Instruct",
    "llama-3.3-70b": "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/Llama-3.3-70B-Instruct",
    "deepseek-3.1-8b": "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-3.3-70b": "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/DeepSeek-R1-Distill-Llama-70B"
}

DATA_PATH = "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/data/106.MIMIC-III Outcome.Diagnosis.SFT.json"

# Load dataset
with open(DATA_PATH, 'r') as f:
    dataset = json.load(f)

def extract_codes(output_text):
    """
    Parses ICD-9 codes from model output like:
    'ICD-9 Diagnosis codes: 428, 486, 584'
    """
    prefix = "ICD-9 Diagnosis codes:"
    if prefix in output_text:
        output_text = output_text.split(prefix)[-1]
    return set(code.strip() for code in output_text.strip().split(",") if code.strip())

def compute_accuracy(preds, golds):
    """
    Compute accuracy based on number of correct codes matched.
    """
    total = len(golds)
    correct = 0
    for pred, gold in zip(preds, golds):
        if gold:  # avoid division by zero
            correct += len(pred & gold) / len(gold)
    return correct / total

def evaluate_model(model_name, model_path, dataset):
    print(f"Evaluating {model_name} using vLLM...")

    # Initialize vLLM with tensor parallelism across 4 GPUs
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=128,
        top_p=1.0,
    )

    predictions = []
    references = []
    results = []

    for example in tqdm(dataset):
        example_instruction = (
    "You are given a patient's hospital admission note.\n\n"
    "Your task is to extract only the **numerical ICD-9 diagnosis codes** relevant to this case.\n\n"
    "Format your response **exactly** as follows:\n"
    "ICD-9 Diagnosis codes: code1, code2, code3, ..., codeN\n\n"
    "- Do NOT include any explanations, summaries, lab values, or patient details.\n"
    "- Return only numerical ICD-9 diagnosis codes (e.g., 428, 486).\n"
    "- Exclude any codes that begin with letters (e.g., E934, V125).\n"
    "- If no diagnosis codes are found, return: ICD-9 Diagnosis codes:"
)

        input_text = f"{example_instruction}\n\nNote:\n{example['input']}\n"

        # Generate using vLLM
        outputs = llm.generate([input_text], sampling_params)
        output_text = outputs[0].outputs[0].text.strip()

        # Try to extract only the new portion after the input
        generated_part = output_text.strip()

        pred_codes = extract_codes(generated_part)
        gold_codes = extract_codes(example["output"])

        predictions.append(pred_codes)
        references.append(gold_codes)

        results.append({
            "id": example.get("id", None),
            "input": example["input"],
            "generated_output": ", ".join(sorted(pred_codes)),
            "original_output": ", ".join(sorted(gold_codes))
        })

    acc = compute_accuracy(predictions, references)
    print(f"{model_name} Accuracy: {acc:.4f}")

    # Save outputs
    output_file = f"inference_outputs_{model_name.replace('/', '_')}_vllm.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved outputs to {output_file}")


if __name__ == "__main__":
    model_to_run = "llama-3.3-70b"
    evaluate_model(model_to_run, MODEL_PATHS[model_to_run], dataset)

    # for name, path in MODEL_PATHS.items():
    #     evaluate_model(name, path, dataset)
