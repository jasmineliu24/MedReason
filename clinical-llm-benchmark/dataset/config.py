import random
import numpy as np

str_template_direct = (
    "Return your answer in the following format. DO NOT GIVE ANY EXPLANATION:"
)
str_template_cot = """Solve it in a step-by-step fashion, return your answer in the following format, PROVIDE DETAILED ANALYSIS BEFORE THE RESULT:
Analysis:
...
Result:"""


def transform_instruction_cot(instruction):
    return instruction.replace(
        str_template_direct,
        str_template_cot,
    )


def transform_instruction_direct(instruction):
    return instruction.replace(
        str_template_cot,
        str_template_direct,
    )


def get_instruction_cot():
    return str_template_cot


def get_instruction_direct():
    return str_template_direct


def extract_cot_pred(str_response):
    list_cot_split_token = ["Result:"]
    for cot_split_token in list_cot_split_token:
        cot_split_token = cot_split_token.lower()
        if cot_split_token in str_response:
            str_response = str_response.split(cot_split_token, 1)
            return str_response[1].strip()
    return str_response


def get_models_evaluate():
    list_model = [
        "gemma-2-9b-it",
        "gemma-2-27b-it",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.3-70B-Instruct",
        "Llama-3.1-Nemotron-70B-Instruct-HF",
        "meditron-7b",
        "meditron-70b",
        "MeLLaMA-13B-chat",
        "MeLLaMA-70B-chat",
        "Llama3-OpenBioLLM-8B",
        "Llama3-OpenBioLLM-70B",
        "MMed-Llama-3-8B",
        "Llama-3.1-8B-UltraMedical",
        "Llama-3-70B-UltraMedical",
        "Ministral-8B-Instruct-2410",
        "Mistral-Small-Instruct-2409",
        "Mistral-Large-Instruct-2411",
        "BioMistral-7B",
        "Phi-3.5-mini-instruct",
        "Phi-3.5-MoE-instruct",
        "Phi-4",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "QwQ-32B-Preview",
        "Athene-V2-Chat",
        "Yi-1.5-9B-Chat-16K",
        "Yi-1.5-34B-Chat-16K",
        "gpt-35-turbo",
        "gpt-4o",
    ]

    return list_model


def get_metrics_clf():
    return ["accuracy", "f1_macro", "f1_micro", "num_failed_ratio"]


def get_metrics_gen():
    return ["bleu", "rouge", "bertscore", "num_failed_ratio"]


def get_metrics_ext():
    return ["f1_subject", "f1_event", "num_failed_ratio"]


def get_metrics_ext_qa():
    return ["exact_match", "overlap_match"]


def get_pred_none_clf(list_pred, list_label):
    # count the number of invalid response
    num_failed = sum([1 for pred in list_pred if pred == -1])
    # get the valid labels
    labels_valid = list(set(list_label))
    # random label
    list_pred = [
        np.random.choice(labels_valid) if pred == -1 else pred for pred in list_pred
    ]

    return list_pred, num_failed


def get_pred_none_clf_mul_label(list_list_pred, list_list_label):
    # count the number of invalid response
    num_failed = sum([1 for pred in list_list_pred if -1 in pred])
    # get the valid labels
    labels_valid = list(
        set([label for list_label in list_list_label for label in list_label])
    )
    # random one label for multi-label classification
    list_list_pred = [
        [np.random.choice(labels_valid)] if -1 in list_pred else list_pred
        for list_pred in list_list_pred
    ]

    return list_list_pred, num_failed


def get_pred_none_clf_mul_question(list_list_pred, list_list_label):
    # count the number of invalid response
    num_failed = sum([1 for list_pred in list_list_pred if -1 in list_pred])
    num_question = len(list_list_label[0])
    # get the unique label for each question
    dict_idx_label = {
        idx: list(set([list_label[idx] for list_label in list_list_label]))
        for idx in range(num_question)
    }

    # random label
    list_list_pred = [
        [
            np.random.choice(dict_idx_label[idx]) if pred == -1 else pred
            for idx, pred in enumerate(list_pred)
        ]
        for list_pred in list_list_pred
    ]

    return list_list_pred, num_failed


def get_pred_none_ext(list_pred):
    # return the empty list
    num_failed = sum([1 for pred in list_pred if -1 in pred])
    list_pred = [[] if -1 in pred else pred for pred in list_pred]

    return list_pred, num_failed


def get_pred_none_gen(list_pred):
    # return the empty string
    num_failed = sum([1 for pred in list_pred if pred == -1])
    list_pred = ["" if pred == -1 else pred for pred in list_pred]

    return list_pred, num_failed


def get_pred_none_gen_qa_mul(list_list_pred):
    # return the empty string
    num_failed = sum([1 for list_pred in list_list_pred if -1 in list_pred])
    list_list_pred = [
        ["" if pred == -1 else pred for pred in list_pred]
        for list_pred in list_list_pred
    ]

    return list_list_pred, num_failed
