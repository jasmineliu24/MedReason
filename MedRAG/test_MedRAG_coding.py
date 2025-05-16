import os
import re
import sys
import json
import shutil
import random
import logging
import argparse
import datetime
import setproctitle
from tqdm import tqdm
from src.medrag import MedRAG
from MIRAGE.src.utils import QADataset, locate_answer

# Assuming clinical-llm-benchmark is a sibling folder to MedRAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'clinical-llm-benchmark')))

# ignore warnings
import warnings

# remove UndefinedMetricWarning
warnings.filterwarnings("ignore")

# self defined functions
from model.init import load_model, seed_everything
from model.inference import run_hf, run_vllm

from dataset.dataset import GeneralTask

from util.tool import init_logger

def batch_data(data_list, batch_size=1):
    batch_data = []
    for i in range(0, len(data_list), batch_size):
        batch_data.append(data_list[i:i + batch_size])
    return batch_data

with open('/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/clinical-llm-benchmark/dataset_raw/106.MIMIC-III Outcome.Diagnosis.SFT.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

questions = []
options = []
correct_answers = []

for data in dataset:
    questions.append(data['instruction'])
    options.append(data['input'])
    correct_answers.append(data['output'])

total_questions = len(questions)
correct_predictions = 0
error_examples = []


# def save_result(args, logger, list_dict_data, list_response):
#     for dict_data, response in zip(list_dict_data, list_response):
#         dict_data["pred"] = response
#     # save into json
#     with open(args.path_file_result, "w", encoding="utf-8") as file:
#         json.dump(list_dict_data, file, indent=4, ensure_ascii=False)
#     logger.info(f"Save: {args.path_file_result}")
#     logger.info("---------------------------------")

    # return list_dict_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        type=str,
        default="1.ADE.SFT.adverse_effect",
        help="The path of the data file, maybe a single task or multiple tasks.",
    )
    parser.add_argument("--rag", action="store_true")
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="vllm",
        help="The inference mode for the model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Meta-Llama-3.1-70B-Instruct",
        help="The name of the LLM model, only one model.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="The gpus for the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="The temperture for the model.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The top_p for the model.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="The top_k for the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch_size for the model.",
    )
    parser.add_argument(
        "--max_token_input",
        type=int,
        default=7 * 1024,
        help="The max_token_input for the model.",
    )
    parser.add_argument(
        "--max_token_output",
        type=int,
        default=1024,
        help="The max_token_output for the model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/DeepSeek-R1-Distill-Llama-8B",
        help="The model path for the model.",
    )
    parser.add_argument("--corpus_name", type=str, default="Textbooks")
    parser.add_argument("--retriever_name", type=str, default="MedCPT")
    parser.add_argument("--HNSW", action="store_true")

    args = parser.parse_args()

    temperature_sample, top_p_sample, top_k_sample = (
        args.temperature,
        args.top_p,
        args.top_k,
    )

    args.gpus = [int(gpu) for gpu in parser.parse_args().gpus.split(",")]

    list_exp_config = [
        # seed = 42
        ("direct", "greedy", 42),
        # ("cot", "greedy", 42),
        # ("direct-5-shot", "greedy", 42),
    ]

    # # New task
    args.tasks = [
        "106.MIMIC-III Outcome.Diagnosis",
        "106.MIMIC-III Outcome.Procedure"
    ]

    args.tasks = args.tasks.split("+") if isinstance(args.tasks, str) else args.tasks
    args.tasks = [task.strip() for task in args.tasks]
    args.tasks.sort(reverse=False)

    num_exp_all = len(args.tasks) * len(list_exp_config)

    # get the model path
    with open(
        "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/clinical-llm-benchmark/dict_model_path.json",
        "r",
        encoding="utf-8",
    ) as f:
        dict_model_path = json.load(f)
    args.model_path = dict_model_path[args.model_name]

    # Record the time of the start
    time_start = datetime.datetime.now()
    # time: YYYY-MM-DD HH-MM-SS
    str_time_start = time_start.strftime("%Y-%m-%d-%H-%M-%S")

    # initialize the model, tokenizer
    tokenizer, model = load_model(args)
    #debugplease
    #saveme
    #thisneedstobefixed

    # for multiple tasks
    for idx_task, task_name in enumerate(args.tasks):
        args.task_name = task_name
        print(f"Task: {task_name}")
        # load the data
        task = GeneralTask(args=args, task_name=task_name)
        # path
        args.path_dir_result = f"result/{task_name}/{args.model_name}/"
        os.makedirs(args.path_dir_result, exist_ok=True)

        # logger
        time_start_task = datetime.datetime.now()
        # time: YYYY-MM-DD HH-MM-SS
        str_time_start_task = time_start_task.strftime("%Y-%m-%d-%H-%M-%S")
        args.path_file_log = os.path.join(
            args.path_dir_result, f"{str_time_start_task}.log"
        )
        logger = init_logger(args.path_file_log)

        # record data and model
        logger.info(f"Model: {args.model_name}: {args.model_path}")
        logger.info(f"Task: {task_name}")
        logger.info(
            f"Size: {len(task.dataset_train)}/{len(task.dataset_val)}/{len(task.dataset_test)}"
        )
        logger.info(f"Start on: {str_time_start_task}")
        logger.info("\n========================================================\n")

        # for multiple experiments
        for idx_exp, (prompt_mode, decoding_strategy, seed) in enumerate(
            list_exp_config
        ):
            num_exp = idx_task * len(list_exp_config) + (idx_exp + 1)
            # set the parameters
            args.prompt_mode = prompt_mode

            # setup the task, especially the prompt mode: "direct", "cot", "n-shot"
            task.setup(tokenizer, prompt_mode)

            # initialize the dataloader
            dataloader = task.dataloader_test()

            # seed
            args.seed = seed
            seed_everything(args.seed)

            # set the decoding strategy and parameters
            args.decoding_strategy = decoding_strategy
            if args.decoding_strategy.lower() == "greedy":
                args.temperature, args.top_p, args.top_k = 0, None, None
            else:
                args.temperature, args.top_p, args.top_k = (
                    temperature_sample,
                    top_p_sample,
                    top_k_sample,
                )

            # genrate other parameters
            args.name_exp = (
                f"{task_name}-{args.prompt_mode}-{args.decoding_strategy}-{args.seed}"
            )
            args.path_file_result = os.path.join(
                args.path_dir_result, f"{args.name_exp}.result.json"
            )

            # setproctitle
            setproctitle.setproctitle(
                f"Benchmark: {args.model_name} {args.name_exp}-({num_exp}/{num_exp_all})"
            )

            # logger.info the name of experiment and time
            logger.info(f"Name of experiment: {args.name_exp}")

            # model config
            for key, value in args.__dict__.items():
                if key != "tasks":
                    logger.info(f"{key}: {value}")

            # Run the main
            if args.inference_mode == "vllm":
                # list_response = run_vllm(
                #     args=args, logger=logger, dataloader=dataloader, model=model
                # )
                save_dict = []

                batched_questions = batch_data(questions, args.batch_size)
                batched_options = batch_data(options, args.batch_size)
                batched_correct_answers = batch_data(correct_answers, args.batch_size)

                medrag = MedRAG(rag=args.rag, retriever_name=args.retriever_name, corpus_name=args.corpus_name, llm_name=args.model_path, HNSW=args.HNSW, db_dir = "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/MedRAG/corpus")
                logger.info("Initiated MedRAG")

                for idx, (question, option, correct_answer) in enumerate(zip(batched_questions, batched_options, batched_correct_answers), start=1):
                    answer_medrag, snippets, scores = medrag.batch_answer(questions=question, options=option, k=5, save_dir="/retrieval_results")
                    logger.info("Completed MedRAG batch answer")

                    for i, (answer) in enumerate(answer_medrag):
                        logger.info("Iterating through each answer in medrag")
                        save_dict.append({"question":question[i], "option":options[i], "generate_text":answer, "answer":correct_answer[i], "snippets": snippets[i]})
                        logger.info("Append result to dictionary")
                        model_answer = answer.split('"answer_choice": "')[-1].strip()
                        model_answer = locate_answer(model_answer)
                        # Print the model's answer for debugging
                        print(f"Model Answer: {model_answer}")
                        logger.info(f"Model Answer: {model_answer}")

                        # Check if the model's answer is correct
                        if model_answer == correct_answer[i].upper():
                            correct_predictions += 1
                        else:
                            error_examples.append((idx * args.batch_size + i, question, correct_answer, model_answer))

                        logger.info("Calculating model accuracy")

                    with open("/model_output.json", 'w') as file:
                        json.dump(save_dict, file)
                        logger.info("Save output into json file")

            else:
                list_response = run_hf(
                    args=args,
                    logger=logger,
                    dataloader=dataloader,
                    model=model,
                    tokenizer=tokenizer,
                )
            # save the result
            # list_dict_data = save_result(args, logger, task.dataset_test, list_response)

            logger.info(f"End of the experiment: {args.name_exp}")
            logger.info("\n========================================================\n")

    # end of the task
    time_end = datetime.datetime.now()
    str_time_end = time_end.strftime("%Y-%m-%d-%H-%M-%S")
    logger.info(f"End on: {str_time_end}")
    time_used = time_end - time_start
    logger.info(f"Time used: {time_used} seconds")
    logger.info("\n========================================================\n")
