import os
import json
import time
import argparse
import setproctitle

from utils import *

if __name__ == "__main__":
    # model config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Llama-3.1-8B-Instruct",
        help="The name of the LLM model, only one model.",
    )
    parser.add_argument(
        "--path_dir_rewrite",
        type=str,
        default="data/rewrite",
        help="The path of the rewrite data.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="The number of starting point.",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=5,
        help="The number of n-grams.",
    )

    args = parser.parse_args()

    # load model
    model, tokenizer, sampling_params = load_model_vllm(args.model_name)

    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    # input and output path
    dict_task_data = list_benbench_tasks(path=args.path_dir_rewrite)
    print("The number of tasks: ", len(dict_task_data))

    # dict_task_data = {
    #     key: value
    #     for key, value in dict_task_data.items()
    #     if key
    #     not in [
    #         # "8.CARES.icd10_block.SFT|test",
    #         # "8.CARES.icd10_sub_block.SFT|test",
    #         # "29.EHRQA.primary_department.SFT|test",
    #         # "29.EHRQA.sub_department.SFT|test",
    #         # "29.EHRQA.qa.SFT|test",
    #         # "33.GOUT-CC.consensus.SFT|test",
    #         # "33.GOUT-CC.predict.SFT|test",
    #         "105.MIMIC-IV CDM.SFT|test",
    #         "106.MIMIC-III Outcome.LoS.SFT|test",
    #         "106.MIMIC-III Outcome.Mortality.SFT|test",
    #         "108.MIMIC-IV DiReCT.PDD.SFT|test",
    #         "108.MIMIC-IV DiReCT.Dis.SFT|test",
    #         "107.MIMIC-IV BHC.SFT|test",
    #     ]
    # }
    # print(" - After filtering: ", len(dict_task_data))

    dict_task_result = {}
    path_dir_result = os.path.join("result/ngrams_gap", args.model_name)
    os.makedirs(path_dir_result, exist_ok=True)

    # The file of the result summary
    path_file_result_all = os.path.join(
        path_dir_result, f"ngram.{args.model_name}.json"
    )

    # record the time of the start
    start_time = time.time()
    str_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time: ", str_start_time)

    for idx_task, (task, data) in enumerate(dict_task_data.items()):
        print("=" * 20, "process task ", task, "=" * 20)

        orgn_data, rewritten_data1, rewritten_data2, rewritten_data3 = data

        # setproctitle
        setproctitle.setproctitle(
            f"Benchmark: Leakage - ngram - {args.model_name} ({idx_task + 1}/{len(dict_task_data)})"
        )

        task_path = os.path.join(path_dir_result, task)
        os.makedirs(task_path, exist_ok=True)

        print(
            f" - Data size: {len(orgn_data)} with {len(rewritten_data1)}, {len(rewritten_data2)}, {len(rewritten_data3)} rewrites"
        )
        ngram_list = []

        path_file_orgn = os.path.join(task_path, f"ngrams-{args.model_name}-orgn.json")
        ngrams_result_orgn = calculate_my_n_gram_accuracy_vllm(
            n=args.ngram,
            k=args.k,
            dataset=orgn_data,
            model=model,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            path_file_output=path_file_orgn,
        )
        print(f"orgn {args.ngram}-gram_acc_avg: ", ngrams_result_orgn["mean_n_grams"])
        ngram_list.append(ngrams_result_orgn["mean_n_grams"])

        # path_file_rewritten1 = os.path.join(
        #     task_path, f"ngrams-{args.model_name}-rewritten1.json"
        # )
        # ngrams_result_rewritten1 = calculate_my_n_gram_accuracy_vllm(
        #     n=args.ngram,
        #     k=args.k,
        #     dataset=rewritten_data1,
        #     model=model,
        #     tokenizer=tokenizer,
        #     sampling_params=sampling_params,
        #     path_file_output=path_file_rewritten1,
        # )
        # print(
        #     f"re-1 {args.ngram}-gram_acc_avg: ",
        #     ngrams_result_rewritten1["mean_n_grams"],
        # )
        # ngram_list.append(ngrams_result_rewritten1["mean_n_grams"])

        # path_file_rewritten2 = os.path.join(
        #     task_path, f"ngrams-{args.model_name}-rewritten2.json"
        # )
        # ngrams_result_rewritten2 = calculate_my_n_gram_accuracy_vllm(
        #     n=args.ngram,
        #     k=args.k,
        #     dataset=rewritten_data2,
        #     model=model,
        #     tokenizer=tokenizer,
        #     sampling_params=sampling_params,
        #     path_file_output=path_file_rewritten2,
        # )
        # print(
        #     f"re-2 {args.ngram}-gram_acc_avg: ",
        #     ngrams_result_rewritten2["mean_n_grams"],
        # )
        # ngram_list.append(ngrams_result_rewritten2["mean_n_grams"])

        # path_file_rewritten3 = os.path.join(
        #     task_path, f"ngrams-{args.model_name}-rewritten3.json"
        # )
        # ngrams_result_rewritten3 = calculate_my_n_gram_accuracy_vllm(
        #     n=args.ngram,
        #     k=args.k,
        #     dataset=rewritten_data3,
        #     model=model,
        #     tokenizer=tokenizer,
        #     sampling_params=sampling_params,
        #     path_file_output=path_file_rewritten3,
        # )
        # print(
        #     f"re-3 {args.ngram}-gram_acc_avg: ",
        #     ngrams_result_rewritten3["mean_n_grams"],
        # )
        # ngram_list.append(ngrams_result_rewritten3["mean_n_grams"])

        print("=" * 20, "process task ", task, "=" * 20)
        dict_task_result[task] = ngram_list

        if idx_task % 2 == 0 or idx_task == len(dict_task_data) - 1:
            # if the file exists, then load the file and update the result
            if os.path.exists(path_file_result_all):
                print("The file exists, load the file and update the result.")
                with open(path_file_result_all, "r", encoding="utf-8") as fp:
                    result_before = json.load(fp)
                    for key, value in dict_task_result.items():
                        if key in result_before and result_before[key]:
                            dict_task_result[key] = result_before[key].extend(value)

            with open(path_file_result_all, "w", encoding="utf-8") as fp:
                json.dump(dict_task_result, fp, ensure_ascii=False, indent=2)

    # record the time of the end
    end_time = time.time()
    str_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End time: ", str_end_time)
    cost_time_hour = (end_time - start_time) / 60
    print("Time cost: ", cost_time_hour, " mins.")
    print("All tasks are done.")
