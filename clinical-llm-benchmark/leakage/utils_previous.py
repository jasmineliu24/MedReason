import os
import regex
import json
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)
    print(f"seed everything: {seed}")


def text_normalize(str_text):
    str_text = regex.sub(r"(\s*\n\s*)+", "\n", str_text)
    str_text = regex.sub(r"[ \t]+", " ", str_text)

    return str_text


list_model_small = [
    "gemma-2-9b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "meditron-7b",
    "MeLLaMA-13B-chat",
    "Llama3-OpenBioLLM-8B",
    "MMed-Llama-3-8B",
    "Llama-3.1-8B-UltraMedical",
    "Ministral-8B-Instruct-2410",
    "Mistral-Small-Instruct-2409",
    "BioMistral-7B",
    "Phi-3.5-mini-instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Yi-1.5-9B-Chat-16K",
]
list_model_mid = ["Phi-4", "gemma-2-27b-it", "QwQ-32B-Preview", "Yi-1.5-34B-Chat-16K"]


def get_num_gpu_for_processor(model_name):
    print("Model name:", model_name)
    if model_name in list_model_small or model_name in list_model_mid:
        print("Small model: one gpu")
        return 1
    else:
        print("Large model: two gpu")
        return 2


def get_batch_size(model_name):
    if model_name in list_model_small:
        return 8
    else:
        return 8


def init_distributed():
    if not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(
            backend="nccl", init_method="env://", rank=local_rank, world_size=world_size
        )

        print(f"[Process {local_rank}] Initialized on GPU {local_rank}")

        return local_rank, world_size
    else:
        print("Distributed environment already initialized")
        return -1, -1


def get_max_token(model_name, model_path):
    # length setting
    if "Qwen" in model_name or "Athene" in model_name.lower():
        path_file_config = os.path.join(model_path, "tokenizer_config.json")
        with open(path_file_config, "r", encoding="utf-8") as f:
            dict_config = json.load(f)
        max_token_all = dict_config["model_max_length"]
    elif "BioMistral-7B" == model_name:
        max_token_all = 2048
    else:
        path_file_config = os.path.join(model_path, "config.json")
        with open(path_file_config, "r", encoding="utf-8") as f:
            dict_config = json.load(f)
        max_token_all = dict_config["max_position_embeddings"]

    return max_token_all


def load_model(model_name):
    print(f"Loaded model: {model_name} with hf")
    with open(
        "/PHShome/jn180/llm_public_host/0-README/dict_model_path.json",
        "r",
        encoding="utf-8",
    ) as f:
        dict_model_path = json.load(f)
    model_path = dict_model_path[model_name]

    # if multi_process:
    #     # 初始化分布式训练
    #     local_rank, world_size = init_distributed()

    #     # 读取可用 GPU
    #     visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    #     visible_gpus = list(map(int, visible_gpus))

    #     # 获取当前进程的 GPU 组
    #     gpus_per_process = get_num_gpu_for_processor(model_name)
    #     start_idx = local_rank * gpus_per_process
    #     used_gpus = visible_gpus[start_idx : start_idx + gpus_per_process]

    #     print(f"[Process {local_rank}] using GPUs: {used_gpus}")

    #     # -------------- 开始加载模型 --------------
    #     # 1) 先只加载模型结构 (device_map="meta") 避免 OOM
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.bfloat16,
    #         low_cpu_mem_usage=True,
    #         device_map="meta",
    #     )

    #     # 2) 构建 device_map
    #     dict_gpu_memory = {start_idx + i: "75GB" for i in range(gpus_per_process)}
    #     devices = dict_gpu_memory.keys()
    #     device_map = infer_auto_device_map(model, max_memory=dict_gpu_memory)

    #     # 3) 重新加载模型到真正的 device_map
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.bfloat16,
    #         device_map=device_map,
    #         low_cpu_mem_usage=True,
    #     )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    model.eval()

    # tokenizer setting
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Tokenizer: Now pad_token_id is:", tokenizer.pad_token_id)
    else:
        print("Tokenizer: pad_token_id is already set:", tokenizer.pad_token_id)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        print("Model: Now pad_token_id is:", model.generation_config.pad_token_id)
    else:
        print(
            "Model: pad_token_id is already set:", model.generation_config.pad_token_id
        )

    max_token_all = get_max_token(model_name, model_path)

    return (model, tokenizer, max_token_all)


def load_model_vllm(model_name, seed=42, ngram=5):
    print(f"Loaded model: {model_name} with vllm")

    # model path
    with open(
        "/PHShome/jn180/llm_public_host/0-README/dict_model_path.json",
        "r",
        encoding="utf-8",
    ) as f:
        dict_model_path = json.load(f)
    model_path = dict_model_path[model_name]
    max_token_all = get_max_token(model_name, model_path)
    num_gpus = os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1
    # model init
    if "mistral" in model_name.lower() and "biomistral" not in model_name.lower():
        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )
    else:
        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
        )
    # sampling params
    sampling_params = SamplingParams(seed=seed, temperature=0, max_tokens=ngram)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    return model, tokenizer, sampling_params


def list_benbench_tasks(path="benbench"):
    file_names = os.listdir(path)
    dict_task_data = {}

    for file_name in tqdm(file_names):
        data_orgn = []
        data_test_1 = []
        data_test_2 = []
        data_test_3 = []
        file_path = os.path.join(path, file_name)
        with open(file_path, "r", encoding="utf8") as fp:
            dict_data = json.load(fp)
        task = file_name.replace(".json", "")
        for _, dict_item in dict_data.items():
            data_orgn.append(dict_item["origin"].strip())
            rewritten_list = dict_item["rewritten"]
            data_test_1.append(rewritten_list[0].strip())
            data_test_2.append(rewritten_list[1].strip())
            data_test_3.append(rewritten_list[2].strip())
        dict_task_data[task] = (data_orgn, data_test_1, data_test_2, data_test_3)

    print(f"Loaded {len(dict_task_data)} tasks")

    return dict_task_data


def load_data_from_jsonl(jsonl_file_name, num_samples=3000):
    if (
        ("SVAMP" in jsonl_file_name)
        or ("MMLU" in jsonl_file_name)
        or ("/MATH/" in jsonl_file_name)
        or ("MetaMath" in jsonl_file_name)
    ):
        with open(jsonl_file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(jsonl_file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]

    selected_samples = random.sample(data, min(num_samples, len(data)))
    print(len(selected_samples))

    ds = {"question": [], "answer": []}

    for item in selected_samples:
        if "rewritten" in jsonl_file_name:
            ds["question"].append(item["rewritten_question"])
            ds["answer"].append(item["rewritten_answer"])
        if ("orgn" in jsonl_file_name) and ("GSM8K" in jsonl_file_name):
            ds["question"].append(item["question"])
            ds["answer"].append(item["answer"])
        if ("orgn" in jsonl_file_name) and ("MATH" in jsonl_file_name):
            # print(jsonl_file_name)
            ds["question"].append(item["problem"])
            ds["answer"].append(item["solution"])

    return ds


def find_subsequence(sequence, subsequence):
    """find subsequence, return -1 if find nothing"""
    for i in range(len(sequence)):
        if sequence[i : i + len(subsequence)] == subsequence:
            return i
    print("Not found\n")
    return -1


def calculate_my_ppl(
    model_name,
    max_token_all,
    dataset,
    model,
    tokenizer,
    path_file_output,
):
    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss()

    for combined_text in tqdm(dataset, total=len(dataset)):
        encoding = tokenizer(
            combined_text,
            return_tensors="pt",
            max_length=max_token_all,
        ).to("cuda")

        # Note: This assumes that you no longer need to account for model-specific maximum sequence lengths
        # or to handle different tokenization strategies for different models as was indicated in the commented-out portion of your provided code.

        encoded_text = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]

        with torch.no_grad():
            out_logits = model(encoded_text, attention_mask=attn_mask).logits

        print("logits", out_logits.shape)
        print("input_ids", encoded_text.shape)
        print("input_ids", encoded_text)

        # Adjusted shift_logits and shift_labels for the entire sequence
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = encoded_text[..., 1:].contiguous()

        # Calculate loss for the entire sequence
        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        loss = loss.mean()
        perplexity = torch.exp(loss).item()
        ppls.append(perplexity)

        samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})

    # Saving the samples and their perplexities to a file
    with open(path_file_output, "w", encoding="utf-8") as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(ppls, len(ppls))

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def calculate_my_ppl_in_batch(
    model_name,
    max_token_all,
    dataset,
    model,
    tokenizer,
    path_file_output,
    batch_size=16,
):
    max_token_all = 3072

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="mean")

    for batch_start in tqdm(
        range(0, len(dataset), batch_size), desc="Calculating Perplexity"
    ):
        # if batch_start < batch_size * 30:
        #     continue

        batch_texts = dataset[batch_start : batch_start + batch_size]

        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_token_all,
            # add_special_tokens=False,
        ).to(device)

        # Note: This assumes that you no longer need to account for model-specific maximum sequence lengths
        # or to handle different tokenization strategies for different models as was indicated in the commented-out portion of your provided code.

        input_ids = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]

        with torch.no_grad():
            try:
                outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
                # shape: (batch, seq_len, vocab_size)
                logits = outputs.logits
            except RuntimeError as e:
                print(f"Skipping batch {batch_start} due to error: {e}")
                continue

        for idx_in_batch in range(len(batch_texts)):
            seq_len = attn_mask[idx_in_batch].sum().item()
            idx_token_not_pad = input_ids.shape[1] - seq_len
            if seq_len < 2:
                perplexity = float("inf")
            else:
                # Adjusted shift_logits and shift_labels for the entire sequence
                shift_logits = logits[idx_in_batch, idx_token_not_pad:-1, :].unsqueeze(
                    0
                )
                shift_labels = input_ids[
                    idx_in_batch, idx_token_not_pad + 1 :
                ].unsqueeze(0)

                # print("logits", logits.shape)
                # print("input_ids", input_ids.shape)
                # print("input_ids", input_ids[idx_in_batch])
                # print("shift_logits", input_ids[idx_in_batch, idx_token_not_pad + 1 :])

                # Calculate loss for the entire sequence
                loss = loss_fct(
                    shift_logits.transpose(1, 2),  # (1, vocab_size, seq_len-1)
                    shift_labels,  # (1, seq_len-1)
                )
                # loss = loss.mean()
                perplexity = torch.exp(loss).item()

            idx_in_dataset = batch_start + idx_in_batch
            if perplexity == float("inf"):
                print(f"Skipping sample {idx_in_dataset} due to zero length")
            else:
                ppls.append(perplexity)
            samples_with_ppl.append(
                {
                    "idx": idx_in_dataset,
                    "text": dataset[idx_in_dataset],
                    "perplexity": perplexity,
                }
            )

    with open(path_file_output, "w", encoding="utf-8") as f:
        for item in samples_with_ppl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def calculate_answer_ppl(datasets, model, tokenizer, device, output_file):
    sep_token = "Answer:"
    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for question, answer in tqdm(
        zip(datasets["question"], datasets["answer"]), total=len(datasets["question"])
    ):
        combined_text = question + " " + sep_token + " " + answer
        encoding = tokenizer(combined_text, return_tensors="pt").to(device)

        if (
            ("chatglm2-6b" in output_file)
            or ("chatglm3-6b" in output_file)
            or ("llama" in output_file and "llama-3" not in output_file)
            or ("Abel" in output_file)
            or ("Mistral" in output_file)
            or ("Orca" in output_file)
            or ("loss" in output_file)
            or ("grok" in output_file)
        ):
            sep_token_ids = tokenizer.encode(sep_token, add_special_tokens=False)
        else:
            sep_token_ids = tokenizer.encode(" " + sep_token, add_special_tokens=False)

        sep_index = find_subsequence(encoding["input_ids"][0].tolist(), sep_token_ids)

        if sep_index != -1:
            encoded_text = encoding["input_ids"]
            attn_mask = encoding["attention_mask"]

            answer_attn_mask = torch.zeros_like(attn_mask)
            answer_attn_mask[:, sep_index + len(sep_token_ids) :] = attn_mask[
                :, sep_index + len(sep_token_ids) :
            ]

            try:
                with torch.no_grad():
                    out_logits = model(encoded_text, attention_mask=attn_mask).logits

                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = encoded_text[..., 1:].contiguous()
                shift_attention_mask = answer_attn_mask[..., 1:].contiguous()

                loss = (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels)
                    * shift_attention_mask
                ).sum(1) / shift_attention_mask.sum(1)
                perplexity = torch.exp(loss).mean().item()
                ppls.append(perplexity)
            except torch.cuda.OutOfMemoryError as e:
                print("Error calculating perplexity: ", e)
                continue

            samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})

        if sep_index == -1:
            print(combined_text)
            print("encoded_text: ", encoding["input_ids"])
            exit

    with open(output_file, "w") as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item) + "\n")

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def calculate_total_ppl(datasets, model, tokenizer, device, output_file):
    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss()

    for question, answer in tqdm(
        zip(datasets["question"], datasets["answer"]), total=len(datasets["question"])
    ):
        combined_text = question + " " + answer
        encoding = tokenizer(combined_text, return_tensors="pt").to(device)

        # Note: This assumes that you no longer need to account for model-specific maximum sequence lengths
        # or to handle different tokenization strategies for different models as was indicated in the commented-out portion of your provided code.

        encoded_text = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]

        with torch.no_grad():
            out_logits = model(encoded_text, attention_mask=attn_mask).logits

        # Adjusted shift_logits and shift_labels for the entire sequence
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = encoded_text[..., 1:].contiguous()

        # Calculate loss for the entire sequence
        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        loss = loss.mean()
        perplexity = torch.exp(loss).item()
        ppls.append(perplexity)

        samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})

    # Saving the samples and their perplexities to a file
    with open(output_file, "w") as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item) + "\n")

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


def prepare_prompt_for_chat_model(prefix_str, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant! Please directly continue my content without extra content such as '...'.",
        },
        {"role": "user", "content": prefix_str},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def calculate_my_n_gram_accuracy(
    n, k, dataset, model, tokenizer, device, output_file, model_type="base"
):
    """
    Calculate n-gram accuracy using a language model with batching.
    :param n: Size of the n-gram to predict.
    :param k: Number of starting points to use for each sample.
    :param datasets: Dataset containing questions and answers.
    :param model: Pre-trained language model.
    :param tokenizer: Tokenizer corresponding to the language model.
    :param device: Device to run the model on.
    :param batch_size: Size of each batch.
    :return: n-gram accuracy.
    """
    # if not tokenizer.pad_token:
    #     if tokenizer.eos_token:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         print("no special token")
    if (
        ("deepseek" in output_file)
        or ("llama" in output_file)
        or ("GPT" in output_file)
        or ("phi" in output_file)
        or ("Baichuan-7B" in output_file)
        or ("Aquila-7B" in output_file)
        or ("Mistral" in output_file)
        or ("loss" in output_file)
    ):
        if not tokenizer.pad_token:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print("set pad done")
            else:
                print("no special token")

    if "GPT" in output_file:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            print("set GPT pad done")

    tokenizer.padding_side = "left"
    if ("Aquila" in output_file) or ("phi" in output_file):
        tokenizer.add_prefix_space = True

    accuracies = []  #

    tokenized_samples = []

    for format_text in dataset:
        tokens = tokenizer.tokenize(format_text)
        tokenized_samples.append(tokens)

    detailed_results = []

    for idx in tqdm(range(0, len(dataset))):
        tokens = tokenized_samples[idx]
        len_tokens = len(tokens)
        sample = tokenizer.convert_tokens_to_string(tokens)
        sample_results = {"idx": idx, "sample": sample, "n_gram_results": []}

        if len_tokens - n - 1 <= 0:
            continue

        sample_correct_n_grams = 0
        sample_total_n_grams = 0
        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file):
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.seq_length) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        elif "chatglm-6b" in output_file:
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.max_sequence_length) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        elif ("Baichuan-13B" in output_file) or ("Baichuan2-13B" in output_file):
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.model_max_length) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        else:
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.max_position_embeddings) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        starting_points = torch.tensor(starting_points)

        for start_index in starting_points:
            prefix_tokens = tokens[:start_index]
            prompt = tokenizer.convert_tokens_to_string(prefix_tokens)
            if model_type == "chat":
                prompt = tokenizer.build_inputs_with_special_tokens(prompt)
            encoding = tokenizer(
                prompt,
                is_split_into_words=False,
                return_tensors="pt",
                padding="longest",
            ).to(device)

            encoding["max_new_tokens"] = n
            encoding["do_sample"] = False

            if (
                ("Mistral" in output_file)
                or ("Abel-7B-002" in output_file)
                or ("deepseek" in output_file)
                or ("phi-2" in output_file)
                or ("loss" in output_file)
                or ("llama-3" in output_file)
            ):
                gens = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id)
            else:
                gens = model.generate(**encoding)

            predicted_ids = gens[0, -n:].tolist()
            original_ids = tokenizer.convert_tokens_to_ids(
                tokens[start_index : start_index + n]
            )

            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            original_text = tokenizer.decode(original_ids, skip_special_tokens=True)

            # Record detailed results
            n_gram_result = {
                "start_index": int(start_index),
                "predicted_text": predicted_text,
                "original_text": original_text,
            }
            sample_results["n_gram_results"].append(n_gram_result)

            sample_total_n_grams += 1
            if original_ids == predicted_ids:
                sample_correct_n_grams += 1

        if sample_total_n_grams > 0:
            sample_accuracy = sample_correct_n_grams / sample_total_n_grams
            accuracies.append(sample_accuracy)

        detailed_results.append(sample_results)

    print(len(accuracies))
    print(accuracies)
    # print(detailed_results)

    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=4)

    return (
        {"n_grams": accuracies, "mean_n_grams": np.mean(accuracies)}
        if accuracies
        else 0
    )


# def calculate_my_n_gram_accuracy_in_batch(
#     model_name,
#     max_token_all,
#     n,
#     k,
#     dataset,
#     model,
#     tokenizer,
#     path_file_output,
#     model_type="base",
#     batch_size=32,
# ):
#     """
#     Calculate n-gram accuracy using a language model with batching.
#     :param model_name: Name of the model.
#     :param n: Size of the n-gram to predict.
#     :param k: Number of starting points to use for each sample.
#     :param datasets: Dataset containing questions and answers.
#     :param model: Pre-trained language model.
#     :param tokenizer: Tokenizer corresponding to the language model.
#     :param batch_size: Size of each batch.
#     :return: n-gram accuracy.
#     """

#     # get the information of the current process
#     local_rank = dist.get_rank()  # rank of the process
#     world_size = dist.get_world_size()  # number of processes

#     if hasattr(model, "hf_device_map"):
#         primary_device = next(iter(model.hf_device_map.values()))
#     else:
#         primary_device = model.device

#     # process the dataset based on the rank
#     total_samples = len(dataset)
#     samples_per_gpu = total_samples // world_size
#     start_idx = local_rank * samples_per_gpu
#     end_idx = (
#         total_samples
#         if local_rank == world_size - 1
#         else (local_rank + 1) * samples_per_gpu
#     )
#     dataset = dataset[start_idx:end_idx]

#     print(
#         f"[Process {local_rank}] Processing samples {start_idx} to {end_idx} (Total: {len(dataset)})"
#     )

#     # Tokenize dataset once to avoid redundancy
#     tokenized_samples = [tokenizer.tokenize(text) for text in dataset]

#     # Initialize results and accuracy tracking
#     detailed_results = []
#     for i, tokens in enumerate(tokenized_samples):
#         reconstructed_text = tokenizer.convert_tokens_to_string(tokens)
#         detailed_results.append(
#             {"idx": start_idx + i, "sample": reconstructed_text, "n_gram_results": []}
#         )

#     accuracies = np.zeros((len(dataset), 2))  # Stores [correct, total] per sample

#     dist.barrier(device_ids=[local_rank])

#     for batch_start in tqdm(
#         range(0, len(dataset), batch_size), desc="Calculating n-gram accuracy"
#     ):
#         current_batch = tokenized_samples[batch_start : batch_start + batch_size]
#         batch_prompts, batch_original_ids, batch_idx_pos = [], [], []

#         # Generate batch prompts
#         for local_idx, tokens in enumerate(current_batch):
#             sample_idx = batch_start + local_idx  # actual index in full dataset
#             len_tokens = len(tokens)

#             # Skip samples that are too short
#             if len_tokens - n - 1 <= 0:
#                 continue

#             # Select starting points using linspace
#             start_positions = np.linspace(
#                 3, min(len_tokens, max_token_all) - n, num=k, dtype=int
#             )

#             # Build prompts and record the original n-gram IDs
#             for start_pos in start_positions:
#                 # Build prefix string
#                 prefix_text = tokenizer.convert_tokens_to_string(tokens[:start_pos])
#                 if model_type == "chat":
#                     # If chat model needs special token building
#                     prefix_text = tokenizer.build_inputs_with_special_tokens(
#                         prefix_text
#                     )

#                 batch_prompts.append(prefix_text)
#                 batch_original_ids.append(
#                     tokenizer.convert_tokens_to_ids(tokens[start_pos : start_pos + n])
#                 )
#                 batch_idx_pos.append((sample_idx, start_pos))

#         if not batch_prompts:
#             continue

#         encodings = tokenizer(
#             batch_prompts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=max_token_all - n,
#         ).to(primary_device)

#         generate_kwargs = {
#             "max_new_tokens": n,
#             "do_sample": False,
#             "temperature": None,
#             "top_k": None,
#             "top_p": None,
#         }

#         with torch.no_grad():
#             outputs = model.generate(**encodings, **generate_kwargs)

#         for i, (sample_idx, start_pos) in enumerate(batch_idx_pos):
#             predicted_ids = outputs[i][-n:].tolist()
#             original_ids = batch_original_ids[i]

#             # Record detailed results
#             predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
#             original_text = tokenizer.decode(original_ids, skip_special_tokens=True)
#             detailed_results[sample_idx]["n_gram_results"].append(
#                 {
#                     "start_index": int(start_pos),
#                     "predicted_text": predicted_text,
#                     "original_text": original_text,
#                 }
#             )

#             if predicted_ids == original_ids:
#                 accuracies[sample_idx][0] += 1  # correct
#             accuracies[sample_idx][1] += 1  # total

#     # Gather results from all processes
#     gathered_accuracies = [None] * world_size
#     gathered_details = [None] * world_size
#     dist.all_gather_object(gathered_accuracies, accuracies)
#     dist.all_gather_object(gathered_details, detailed_results)

#     final_acc = []
#     for proc_acc in gathered_accuracies:
#         for correct, total in proc_acc:
#             if total > 0:
#                 final_acc.append(correct / total)

#     if local_rank == 0:
#         with open(path_file_output, "w", encoding="utf-8") as f:
#             json.dump(gathered_details, f, indent=4, ensure_ascii=False)

#     return {"n_grams": gathered_accuracies, "mean_n_grams": np.mean(final_acc)}


def calculate_my_n_gram_accuracy_hf_batch(
    n,
    k,
    dataset,
    model,
    tokenizer,
    model_name,
    path_file_output,
    token_gap=5,
    batch_size=32,
    model_type="chat",
    min_prompt_start=10,
    max_token_all=3072,
):
    """
    Calculate n-gram accuracy using Hugging Face model in batches.
    For each sample, select multiple starting positions (using a series based on token_gap
    and additional preset positions), build prompts, let the model generate an n-gram,
    and then compare the generated n-gram with the reference.

    Args:
        n (int): Size of the n-gram to predict.
        k (int): Number of main starting points (using min_prompt_start + i*token_gap).
        dataset (list[str]): List of text samples.
        model: Hugging Face model with a .generate() method.
        tokenizer: Corresponding tokenizer.
        sampling_params (dict): Sampling parameters for model.generate() call.
        path_file_output (str): Output path for saving detailed JSON results.
        token_gap (int): Gap between consecutive starting positions. default 5.
        min_prompt_start (int): Minimum starting token index. default 10.
        max_token_all (int): Maximum token positions to consider from each sample. default 3072.
        batch_size (int): Batch size.

    Returns:
        dict:
            {
                "n_grams": detailed_results,  # Detailed info about each sample
                "mean_n_grams": float,        # Mean accuracy across valid positions
            }
    """
    max_token_all = 3072

    generate_kwargs = {
        "max_new_tokens": n,
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize dataset once to avoid redundancy
    dataset_tokenized = [
        tokenizer.tokenize(text)
        for text in dataset
        # , add_special_tokens=False
    ]

    # Initialize results and accuracy tracking
    detailed_results = [
        {"idx": i, "sample": text, "n_gram_results": []}
        for i, text in enumerate(dataset)
    ]

    accuracies = np.zeros((len(dataset), 2))  # Stores [correct, total] per sample

    list_input = []
    list_output_text, list_output_token, list_idx_pos = [], [], []

    preset_positions = [40, 80, 160, 320, 640]

    # iterate over all samples
    for idx_data, tokens in enumerate(dataset_tokenized):
        # if the sample is too short, skip
        if len(tokens) < min_prompt_start + (k - 1) * token_gap + n:
            continue
        # generate starting positions based on token_gap
        list_pos_start = [min_prompt_start + i * token_gap for i in range(k)]
        # supplement with preset positions, if possible
        for pos in preset_positions:
            if pos + n <= len(tokens):
                list_pos_start.append(pos)
        # deduplicate and sort the starting positions
        list_pos_start = list(set(list_pos_start))
        # Build prompts and record the original n-gram IDs
        for pos_start in list_pos_start:
            prefix_text = tokenizer.convert_tokens_to_string(tokens[:pos_start])
            output_text = tokenizer.convert_tokens_to_string(
                tokens[pos_start : pos_start + n]
            )
            output_token = tokenizer.convert_tokens_to_ids(
                tokens[pos_start : pos_start + n]
            )
            list_input.append(prefix_text)
            list_output_text.append(output_text)
            list_output_token.append(output_token)
            list_idx_pos.append((idx_data, pos_start))

    # Generate responses in batches
    num_batches = (len(list_input) + batch_size - 1) // batch_size
    list_pred_token = []
    list_pred_text = []
    for i in tqdm(range(num_batches), desc="Generating n-gram predictions"):
        batch_inputs = list_input[i * batch_size : (i + 1) * batch_size]
        # Tokenize the batch and generate predictions
        batch_encoded = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_token_all - n,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**batch_encoded, **generate_kwargs)
        # Decode the predictions and store them
        for j in range(outputs.shape[0]):
            pred_ids = outputs[j][-n:].tolist()
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            list_pred_token.append(pred_ids)
            list_pred_text.append(pred_text)

    # Iterate over the predictions and compare them to the original n-grams
    for i, (sample_idx, pos_start) in enumerate(list_idx_pos):
        pred_token = list_pred_token[i]
        pred_text = list_pred_text[i]
        orig_token = list_output_token[i]
        orig_text = list_output_text[i]
        acc_binary = pred_token == orig_token
        acc_prop = sum(p == o for p, o in zip(pred_token, orig_token)) / len(pred_token)
        detailed_results[sample_idx]["n_gram_results"].append(
            {
                "start_index": int(pos_start),
                "prompt": list_input[i],
                "original_ids": orig_token,
                "original_text": orig_text,
                "predicted_ids": pred_token,
                "predicted_text": pred_text,
                "accuracy_binary": acc_binary,
                "accuracy_proportion": acc_prop,
            }
        )
        # Update the accuracy tracking
        if acc_binary:
            accuracies[sample_idx][0] += 1
        accuracies[sample_idx][1] += 1

    # Calculate the mean accuracy across all samples
    list_acc = [correct / total for correct, total in accuracies if total > 0]
    mean_acc = np.mean(list_acc) if list_acc else 0.0

    # Save the detailed results
    with open(path_file_output, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    return {"n_grams": detailed_results, "mean_n_grams": mean_acc}


def calculate_my_n_gram_accuracy_vllm(
    n,
    k,
    dataset,
    model,
    tokenizer,
    sampling_params,
    path_file_output,
    token_gap=5,
    min_prompt_start=10,
    max_token_all=3072,
):
    """
    Calculate n-gram accuracy using a vLLM model interface. For each text sample,
    we select multiple starting positions, build a prompt, let the model generate
    an n-gram, and compare it to the reference n-gram.

    Args:
        n (int): Size of the n-gram to predict.
        k (int): Number of main starting points to use for each sample.
        dataset (list[str]): List of text samples.
        model: vLLM model interface with a .generate() method.
        tokenizer: Corresponding tokenizer with .tokenize() and .convert_tokens_to_string().
        sampling_params: Sampling params for the model.generate() call.
        path_file_output (str): Output path to save detailed JSON results.
        token_gap (int): Gap (in tokens) between consecutive starting points. Default 5.
        min_prompt_start (int): Minimum starting token index. Default 10.
        max_token_all (int): Maximum token positions to consider from each sample. Default 3072.

    Returns:
        dict:
            {
                "n_grams": detailed_results,  # Detailed info about each sample
                "mean_n_grams": float,        # Mean accuracy across valid positions
            }
    """

    # Preprocess the dataset, merge the redundant spaces, \n, \t
    # dataset = [text_normalize(text) for text in dataset]

    # Prepare the dataset
    dataset_tokenized = [
        tokenizer.tokenize(text, add_special_tokens=False) for text in dataset
    ]
    detailed_results = [
        {"idx": i, "sample": text, "n_gram_results": []}
        for i, text in enumerate(dataset)
    ]

    # Initialize results and accuracy tracking
    accuracies = np.zeros((len(dataset), 2))  # [correct, total]

    list_input = []
    list_output_text, list_output_token, list_idx_pos = [], [], []

    preset_positions = [40, 80, 160, 320, 640]

    for idx_data, tokens in enumerate(dataset_tokenized):
        # len_tokens = len(tokens)
        # if min_prompt_start + n > len_tokens:
        #     continue
        # Select starting points using linspace
        # list_pos_start = np.linspace(
        #     min_prompt_start,
        #     min(len_tokens, max_token_all) - n,
        #     num=k,
        #     dtype=int,
        # )
        # list_pos_start = list(set(list_pos_start))

        if len(tokens) < min_prompt_start + (k - 1) * token_gap + n:
            continue

        list_pos_start = [min_prompt_start + i * token_gap for i in range(k)]
        for pos_start in preset_positions:
            if pos_start + n <= len(tokens):
                list_pos_start.append(pos_start)

        # Build prompts and record the original n-gram IDs
        for pos_start in list_pos_start:
            prefix_text = tokenizer.convert_tokens_to_string(tokens[:pos_start])
            output_text = tokenizer.convert_tokens_to_string(
                tokens[pos_start : pos_start + n]
            )
            output_token = tokenizer.convert_tokens_to_ids(
                tokens[pos_start : pos_start + n]
            )
            list_input.append(prefix_text)
            list_output_text.append(output_text)
            list_output_token.append(output_token)
            list_idx_pos.append((idx_data, pos_start))

    # Generate responses
    list_response_generator = model.generate(
        list_input,
        sampling_params,
        use_tqdm=True,
    )
    # Extract the predicted texts
    list_pred_token = [
        list(response.outputs[0].token_ids[-n:]) for response in list_response_generator
    ]
    list_pred_text = [response.outputs[0].text for response in list_response_generator]

    for i, (idx_data, pos_start) in enumerate(list_idx_pos):
        # Extract the predicted and original n-grams
        pred_token = list_pred_token[i]
        pred_text = list_pred_text[i]
        output_token = list_output_token[i]
        output_text = list_output_text[i]
        acc_binary = pred_token == output_token
        acc_prop = sum(p == o for p, o in zip(pred_token, output_token)) / len(
            pred_token
        )
        # Record detailed results
        detailed_results[idx_data]["n_gram_results"].append(
            {
                "start_index": int(pos_start),
                "prompt": list_input[i],
                "original_ids": output_token,
                "original_text": output_text,
                "predicted_ids": pred_token,
                "predicted_text": pred_text,
                "accuracy_binary": acc_binary,
                "accuracy_proportion": acc_prop,
            }
        )
        if acc_binary:
            accuracies[idx_data][0] += 1  # correct
        accuracies[idx_data][1] += 1  # total

    list_acc = [correct / total for correct, total in accuracies if total > 0]
    mean_acc = np.mean(list_acc) if list_acc else 0.0

    # Save the detailed results
    with open(path_file_output, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    return {"n_grams": detailed_results, "mean_n_grams": mean_acc}


def calculate_n_gram_accuracy(
    n, k, dataset, model, tokenizer, device, output_file, model_type="base"
):
    """
    Calculate n-gram accuracy using a language model with batching.
    :param n: Size of the n-gram to predict.
    :param k: Number of starting points to use for each sample.
    :param datasets: Dataset containing questions and answers.
    :param model: Pre-trained language model.
    :param tokenizer: Tokenizer corresponding to the language model.
    :param device: Device to run the model on.
    :param batch_size: Size of each batch.
    :return: n-gram accuracy.
    """
    # if not tokenizer.pad_token:
    #     if tokenizer.eos_token:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         print("no special token")
    if (
        ("deepseek" in output_file)
        or ("llama" in output_file)
        or ("GPT" in output_file)
        or ("phi" in output_file)
        or ("Baichuan-7B" in output_file)
        or ("Aquila-7B" in output_file)
        or ("Mistral" in output_file)
        or ("loss" in output_file)
    ):
        if not tokenizer.pad_token:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print("set pad done")
            else:
                print("no special token")

    if "GPT" in output_file:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            print("set GPT pad done")

    tokenizer.padding_side = "left"
    if ("Aquila" in output_file) or ("phi" in output_file):
        tokenizer.add_prefix_space = True

    accuracies = []  #

    tokenized_samples = []

    for question, answer in zip(dataset["question"], dataset["answer"]):
        if (
            ("hellaswag" in output_file)
            or ("Truthful" in output_file)
            or ("MMLU" in output_file)
        ):
            format_text = f"{question}{answer}"
        else:
            format_text = f"{question} {answer}"
        tokens = tokenizer.tokenize(format_text)
        tokenized_samples.append(tokens)

    detailed_results = []

    for idx in tqdm(range(0, len(dataset["question"]))):
        tokens = tokenized_samples[idx]
        len_tokens = len(tokens)
        sample = tokenizer.convert_tokens_to_string(tokens)
        sample_results = {"idx": idx, "sample": sample, "n_gram_results": []}

        if len_tokens - n - 1 <= 0:
            continue

        sample_correct_n_grams = 0
        sample_total_n_grams = 0
        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file):
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.seq_length) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        elif "chatglm-6b" in output_file:
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.max_sequence_length) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        elif ("Baichuan-13B" in output_file) or ("Baichuan2-13B" in output_file):
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.model_max_length) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        else:
            starting_points = np.linspace(
                2,
                min(len_tokens, model.config.max_position_embeddings) - n,
                num=k,
                endpoint=True,
                dtype=int,
            )
        starting_points = torch.tensor(starting_points)

        for start_index in starting_points:
            prefix_tokens = tokens[:start_index]
            prompt = tokenizer.convert_tokens_to_string(prefix_tokens)
            if model_type == "chat":
                prompt = tokenizer.build_inputs_with_special_tokens(prompt)
            encoding = tokenizer(
                prompt,
                is_split_into_words=False,
                return_tensors="pt",
                padding="longest",
            ).to(device)

            encoding["max_new_tokens"] = n
            encoding["do_sample"] = False

            if (
                ("Mistral" in output_file)
                or ("Abel-7B-002" in output_file)
                or ("deepseek" in output_file)
                or ("phi-2" in output_file)
                or ("loss" in output_file)
                or ("llama-3" in output_file)
            ):
                gens = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id)
            else:
                gens = model.generate(**encoding)

            predicted_ids = gens[0, -n:].tolist()
            original_ids = tokenizer.convert_tokens_to_ids(
                tokens[start_index : start_index + n]
            )

            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            original_text = tokenizer.decode(original_ids, skip_special_tokens=True)

            # Record detailed results
            n_gram_result = {
                "start_index": int(start_index),
                "predicted_text": predicted_text,
                "original_text": original_text,
            }
            sample_results["n_gram_results"].append(n_gram_result)

            sample_total_n_grams += 1
            if original_ids == predicted_ids:
                sample_correct_n_grams += 1

        if sample_total_n_grams > 0:
            sample_accuracy = sample_correct_n_grams / sample_total_n_grams
            accuracies.append(sample_accuracy)

        detailed_results.append(sample_results)

    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=4)

    return (
        {"n_grams": accuracies, "mean_n_grams": np.mean(accuracies)}
        if accuracies
        else 0
    )
