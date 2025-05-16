import os
import json
import time
import argparse
import setproctitle

from leakage.utils import *

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
        "--batch_size",
        type=int,
        default="4",
        help="The batch_size for the model.",
    )

    args = parser.parse_args()

    # set seed
    seed_everything(42)

    # load model
    model, tokenizer, max_token_all = load_model(args.model_name)

    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    # input and output path
    dict_task_data = list_benbench_tasks(path=args.path_dir_rewrite)
    print("The number of tasks: ", len(dict_task_data))

    dict_task_result = {}
    path_dir_result = os.path.join("result/ppl", args.model_name)
    os.makedirs(path_dir_result, exist_ok=True)

    # The file of the result summary
    path_file_result_all = os.path.join(path_dir_result, f"ppl.{args.model_name}.json")

    # record the time of the start
    start_time = time.time()
    str_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time: ", str_start_time)

    for idx_task, (task, data) in enumerate(dict_task_data.items()):
        print("=" * 20, "process task ", task, "=" * 20)

        orgn_data, rewritten_data1, rewritten_data2, rewritten_data3 = data

        # setproctitle
        setproctitle.setproctitle(
            f"Benchmark: Leakage - ppl - {args.model_name} ({idx_task + 1}/{len(dict_task_data)})"
        )

        task_path = os.path.join(path_dir_result, task)
        os.makedirs(task_path, exist_ok=True)

        print(
            f" - Data size: {len(orgn_data)} with {len(rewritten_data1)}, {len(rewritten_data2)}, {len(rewritten_data3)} rewrites"
        )

        ppl_list = []

        path_file_orgn = os.path.join(task_path, f"ppl-{args.model_name}-orgn.jsonl")
        ppl_result_orgn = calculate_my_ppl_in_batch(
            model_name=args.model_name,
            max_token_all=max_token_all,
            dataset=orgn_data,
            model=model,
            tokenizer=tokenizer,
            path_file_output=path_file_orgn,
            batch_size=args.batch_size,
        )
        print(f"orgn average_ppl_accuracy: ", ppl_result_orgn["mean_perplexity"])
        ppl_list.append(ppl_result_orgn["mean_perplexity"])

        path_file_rewritten1 = os.path.join(
            task_path, f"ppl-{args.model_name}-rewritten1.jsonl"
        )
        ppl_result_rewritten1 = calculate_my_ppl_in_batch(
            model_name=args.model_name,
            max_token_all=max_token_all,
            dataset=rewritten_data1,
            model=model,
            tokenizer=tokenizer,
            path_file_output=path_file_rewritten1,
            batch_size=args.batch_size,
        )
        print(
            f"rewritten1 average_ppl_accuracy: ",
            ppl_result_rewritten1["mean_perplexity"],
        )
        ppl_list.append(ppl_result_rewritten1["mean_perplexity"])

        path_file_rewritten2 = os.path.join(
            task_path, f"ppl-{args.model_name}-rewritten2.jsonl"
        )
        ppl_result_rewritten2 = calculate_my_ppl_in_batch(
            model_name=args.model_name,
            max_token_all=max_token_all,
            dataset=rewritten_data2,
            model=model,
            tokenizer=tokenizer,
            path_file_output=path_file_rewritten2,
            batch_size=args.batch_size,
        )
        print(
            f"rewritten2 average_ppl_accuracy: ",
            ppl_result_rewritten2["mean_perplexity"],
        )
        ppl_list.append(ppl_result_rewritten2["mean_perplexity"])

        path_file_rewritten3 = os.path.join(
            task_path, f"ppl-{args.model_name}-rewritten3.jsonl"
        )
        ppl_result_rewritten3 = calculate_my_ppl_in_batch(
            model_name=args.model_name,
            max_token_all=max_token_all,
            dataset=rewritten_data3,
            model=model,
            tokenizer=tokenizer,
            path_file_output=path_file_rewritten3,
            batch_size=args.batch_size,
        )
        print(
            f"rewritten3 average_ppl_accuracy: ",
            ppl_result_rewritten3["mean_perplexity"],
        )
        ppl_list.append(ppl_result_rewritten3["mean_perplexity"])

        print("=" * 20, "process task ", task, "=" * 20)
        dict_task_result[task] = ppl_list

        if idx_task % 2 == 0 or idx_task == len(dict_task_data) - 1:
            # if the file exists, then load the file and update the result
            if os.path.exists(path_file_result_all):
                print("The file exists, load the file and update the result.")
                with open(path_file_result_all, "r", encoding="utf-8") as fp:
                    result_before = json.load(fp)
                    dict_task_result.update(result_before)

            with open(path_file_result_all, "w", encoding="utf-8") as fp:
                json.dump(dict_task_result, fp, ensure_ascii=False, indent=2)

    # record the time of the end
    end_time = time.time()
    str_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End time: ", str_end_time)
    cost_time_hour = (end_time - start_time) / 60
    print("Time cost: ", cost_time_hour, " mins.")
    print("All tasks are done.")
