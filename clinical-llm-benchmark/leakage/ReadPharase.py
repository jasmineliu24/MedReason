import re
import os
import json
from tqdm import tqdm


def split_id(id_raw):
    return id_raw.rsplit("|", 1)[0]


def read_single_input_file(file_path):
    file_dict = {}
    with open(file_path, "r", encoding="utf8") as fp:
        for line in fp.readlines():
            item_dict = json.loads(line)
            item_id = split_id(item_dict["custom_id"])
            orgn_content = item_dict["body"]["messages"][1]["content"]
            file_dict[item_id] = orgn_content

    # idx = 0
    # for k, v in file_dict.items():
    #     print(k, v)
    #     idx += 1
    #     if idx == 10:
    #         break
    return file_dict


def read_llm_input_json_files(dir_path):
    all_item_dict = {}
    files = os.listdir(dir_path)
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        items_dict = read_single_input_file(file_path)
        all_item_dict.update(items_dict)
    return all_item_dict


def read_single_output_file(file_path):
    file_dict = {}
    err_cnt = 0
    with open(file_path, "r", encoding="utf8") as fp:
        for line in fp.readlines():
            item_dict = json.loads(line)
            item_id = split_id(item_dict["custom_id"])
            try:
                result_content = item_dict["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
            except:
                err_cnt += 1
                # print(result_data['response']['body']['choices'][0]['message'])
                continue
            if item_id not in file_dict:
                file_dict[item_id] = []
            file_dict[item_id].append(result_content)

    # idx = 0
    # for k, v in file_dict.items():
    #     print(k, v)
    #     idx += 1
    #     if idx == 10:
    #         break
    print(f"read output, output len: {len(file_dict)}, err len: {err_cnt}.")
    return file_dict


def parse_contents(raw_content_list):
    pattern = r"<Output>(.*?)</Output>"
    parsed_list = []
    for raw_content in raw_content_list:
        match = re.search(pattern, raw_content, re.DOTALL)
        if match:
            parsed_list.append(match.group(1).strip())
    return parsed_list


def read_llm_output_jsonl_files(dir_path):
    all_item_dict = {}
    files = os.listdir(dir_path)
    for file_name in files:
        file_path = os.path.join(dir_path, file_name)
        items_dict = read_single_output_file(file_path)
        all_item_dict.update(items_dict)
    return all_item_dict


def cal_ratio(len1, len2):
    return len2 / len1


def filt_rewritten_result(orgn_text, rewritten_list):
    orgn_len = len(orgn_text)
    if orgn_len < 30:
        return True

    # print(orgn_len, len(rewritten_list[0]), len(rewritten_list[1]), len(rewritten_list[2]))
    ratio1 = cal_ratio(orgn_len, len(rewritten_list[0]))
    # print(ratio1)
    if ratio1 < 0.5 or ratio1 > 2:
        return False

    ratio2 = cal_ratio(orgn_len, len(rewritten_list[1]))
    # print(ratio2)
    if ratio2 < 0.5 or ratio2 > 2:
        return False

    ratio3 = cal_ratio(orgn_len, len(rewritten_list[2]))
    # print(ratio3)
    if ratio3 < 0.5 or ratio3 > 2:
        return False
    return True


def merge_input_ouput(intput_dict, output_dict):
    merged_dict = {}
    count_dict = {}

    for item_id, rewritten_text_list in tqdm(output_dict.items()):
        task_id_set = split_id(item_id)

        if task_id_set not in merged_dict:
            merged_dict[task_id_set] = {}
        if task_id_set not in count_dict:
            count_dict[task_id_set] = [0, 0, 0, 0]

        count_dict[task_id_set][0] += 1
        orgn_text = intput_dict.get(item_id, "")
        if orgn_text == "":
            count_dict[task_id_set][1] += 1
            continue

        rewritten_list = parse_contents(rewritten_text_list)
        if len(rewritten_list) != 3:
            count_dict[task_id_set][2] += 1
            continue

        isValid = filt_rewritten_result(orgn_text, rewritten_list)
        if not isValid:
            count_dict[task_id_set][3] += 1
            continue

        merged_dict[task_id_set][item_id] = {
            "origin": orgn_text,
            "rewritten": rewritten_list,
        }
    for k, v in count_dict.items():
        print(k, v)
    return merged_dict


def export_results(dir_path, data):
    for file_id, value in data.items():
        file_path = os.path.join(dir_path, file_id + ".json")
        with open(file_path, "w", encoding="utf8") as fp:
            json.dump(value, fp, ensure_ascii=False, indent=2)


def main():
    input_dir_path = "data/input_train"
    intput_dict = read_llm_input_json_files(input_dir_path)
    output_dir_path = "data/output_train"
    output_dict = read_llm_output_jsonl_files(output_dir_path)
    assert len(intput_dict) == len(output_dict)
    print(f"input len: {len(intput_dict)}, output len: {len(output_dict)}")
    merged_dict = merge_input_ouput(intput_dict, output_dict)
    benbench_dir_path = "data/rewrite_train"
    for k, v in merged_dict.items():
        print(k, len(v))
    export_results(benbench_dir_path, merged_dict)


if __name__ == "__main__":
    main()
