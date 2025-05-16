import os
import re
import json
import numpy as np

from leakage.utils import *

seed_everything(42)


def list_sft(sft_path="sft"):
    sft_files = os.listdir(sft_path)
    list_path_task = []
    for file_name in sft_files:
        if ".json" not in file_name:
            continue
        task_name = file_name.replace(".json", "")
        file_path = os.path.join(sft_path, file_name)
        list_path_task.append([file_path, task_name])
    # sort by task name
    list_path_task = sorted(list_path_task, key=lambda x: x[1])
    return list_path_task


def read_sft(sft_file_path):
    fp = open(sft_file_path, "r", encoding="utf8")
    sft_data = json.load(fp)
    fp.close()
    return sft_data


def truncate_text(text, language, max_words):
    """
    Truncates the input text to the specified maximum number of words while preserving complete sentences.

    Inputs:
    - text (str): The input text to be truncated.
    - language (str): The language of the text. Either 'en' (English), 'zh' (Chinese), or 'jp' (Japanese).
    - max_words (int): The maximum number of words to keep in the truncated text.

    Returns:
    - truncated_text (str): The truncated text with complete sentences.
    """
    # Define the sentence pattern based on the language
    # sentence_pattern = r'.+?[。！？.!?]'
    sentence_endings = r"[。！？.!?]\s*|\n+"
    sentence_pattern = r".+?(?:" + sentence_endings + "|$)"
    sentences = re.findall(sentence_pattern, text)

    # If no sentences are found, assume the input text is a single sentence
    if not sentences:
        sentences = [text]

    truncated_sentences = []
    word_count = 0

    list_languge_no_split = ["cn", "jp", "zh", "ja"]
    for sentence in sentences:
        # Calculate the number of words in the sentence
        num_words_in_sentence = (
            len(sentence)
            if language in list_languge_no_split
            else len(sentence.split())
        )

        if word_count + num_words_in_sentence <= max_words:
            truncated_sentences.append(sentence)
            word_count += num_words_in_sentence
        else:
            # If the sentence itself exceeds the maximum word count, truncate it
            break

    # Join the truncated sentences to form the final text
    truncated_text = "".join(truncated_sentences)
    return truncated_text, word_count


def process_sft(
    file_path,
    task_name,
    split="test",
    num_words=512,
    num_paraphare=3,
    max_sample_num=50000,
):

    instruction = """Please act as a experienced medical text rewriter to paraphrase the following medical text in its original language, following the instructions below:
1. Paraphrase the text by rewording it with new expressions and sentence structures.
2. Do not change the essence of the text.
3. Ensure that you do not to deviate too much from the original content, and try to maintain the same style as much as possible.
4. Provide a clear output format to distinguish between the three rewrites.
Ouput format:
<Output>
    Your rewritten text
</Output>

Please strictly follow the output format. Make sure the output is in the correct format, so that the result can be directly parsed by the evaluation script."""

    sft_data = read_sft(file_path)
    sft_data_test = [item for item in sft_data if item["split"] == "test"]
    if split == "train":
        sft_data_train = [item for item in sft_data if item["split"] == "train"]
        # Extract the same amount of data from the training set as from the test set
        if len(sft_data_train) > len(sft_data_test):
            sft_data = np.random.choice(
                sft_data_train, len(sft_data_test), replace=False
            )
        else:
            sft_data = sft_data_train
    else:
        sft_data = sft_data_test

    lang = sft_data[0]["language"] if "language" in sft_data[0] else "en"

    data_batch = []
    tot_cnt = 0

    for idx_item, sft_item in enumerate(sft_data):
        input_text = sft_item["input"]
        input_text_truncated, word_cnt = truncate_text(input_text, lang, num_words)
        for rewritten_id in range(1, num_paraphare + 1):
            item_dict = {
                "custom_id": f"{task_name}|{split}|{idx_item}|{rewritten_id}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-batch",
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": input_text_truncated},
                    ],
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    # "max_tokens": 1024,
                },
            }
            data_batch.append(item_dict)
            tot_cnt += word_cnt
        if len(data_batch) >= max_sample_num * num_paraphare:
            break
    print(
        file_path,
        "average word cnt: ",
        tot_cnt / len(data_batch),
        "number of data: ",
        len(data_batch),
    )
    return data_batch


def export_data(
    export_dict,
    data_batch,
    is_last,
    split,
    output_dir="llm_input_jsonl",
    max_export_num=100000,
    max_file_size=180,
):
    os.makedirs(output_dir, exist_ok=True)

    file_idx = export_dict["file_idx"]
    export_list = export_dict["export_list"]
    file_size = export_dict["file_size"]  # 追踪当前文件的大小

    # Estimate the size of data_batch
    data_batch_size = sum(
        len(json.dumps(item, ensure_ascii=False).encode("utf-8")) / 1024 / 1024
        for item in data_batch
    )

    # Check if the current data_batch can be added to export_list
    if (
        len(export_list) + len(data_batch) > max_export_num
        or file_size + data_batch_size > max_file_size
    ):
        # If export_list is not empty, export it
        if export_list:
            output_path = os.path.join(
                output_dir, f"rewritten_input_jsonl_{file_idx}.{split}.jsonl"
            )
            with open(output_path, "w", encoding="utf-8") as fp:
                for line in export_list:
                    fp.write(json.dumps(line, ensure_ascii=False) + "\n")

            print(
                f"Exported {len(export_list)} items to {output_path} (Size: {file_size:.2f} MB)"
            )

            # Update export_dict state
            file_idx += 1
            export_list = []
            file_size = 0

    # Add data_batch to export_list
    export_list.extend(data_batch)
    file_size += data_batch_size

    # If this is the last batch, ensure final state is consistent
    if is_last and export_list:
        output_path = os.path.join(
            output_dir, f"rewritten_input_jsonl_{file_idx}.{split}.jsonl"
        )
        with open(output_path, "w", encoding="utf-8") as fp:
            for line in export_list:
                fp.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(
            f"Final export {len(export_list)} items to {output_path} (Size: {file_size:.2f} MB)"
        )

    # Update export_dict state
    export_dict["file_idx"] = file_idx
    export_dict["export_list"] = export_list
    export_dict["file_size"] = file_size


def main():
    # num_samples = 200
    num_words = 512
    split = "train"
    sft_path = "/netapp2/home/jw1399/clinical_text_dataset/clinical-llm-benchmark/dataset_fine/all"
    output_dir = "data/input/"
    os.makedirs(output_dir, exist_ok=True)
    list_path_task = list_sft(sft_path=sft_path)
    export_dict = {"file_idx": 0, "export_list": [], "file_size": 0}
    tot_len = 0
    for idx, (file_path, task_name) in enumerate(list_path_task):
        print(f"Processing task {task_name} ({idx + 1}/{len(list_path_task)})")
        last = idx == len(list_path_task) - 1
        data_batch = process_sft(file_path, task_name, split, num_words)
        tot_len += len(data_batch)
        export_data(export_dict, data_batch, last, split, output_dir)
    print("tol number of data:", tot_len)


if __name__ == "__main__":
    main()
