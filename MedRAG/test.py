from src.medrag import MedRAG
import json
from tqdm import tqdm
import os
import re
import sys
MAX_INT = sys.maxsize
import argparse

sys.path.append("./MIRAGE")
from MIRAGE.src.utils import QADataset, locate_answer
# from utils import QADataset,  locate_answer4pub_llama

# def batch_data(data_list, batch_size=1):
#     n = len(data_list) // batch_size
#     batch_data = []
#     for i in range(n-1):
#         start = i * batch_size
#         end = (i+1)*batch_size
#         batch_data.append(data_list[start:end])

#     last_start = (n-1) * batch_size
#     last_end = MAX_INT
#     batch_data.append(data_list[last_start:last_end])
#     return batch_data

def batch_data(data_list, batch_size=512):
    batch_data = []
    for i in range(0, len(data_list), batch_size):
        batch_data.append(data_list[i:i + batch_size])
    return batch_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="/home/zentek/Qwen2.5-7B-Instruct")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--HNSW", action="store_true")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--corpus_name", type=str, default="Textbooks")
    parser.add_argument("--retriever_name", type=str, default="MedCPT")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--dataset_name", type=str, default="mmlu")
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()

    # rag = True
    # print(args)
    # import pdb; pdb.set_trace()
    if args.rag:
        save_dir = args.results_dir + '/' + args.dataset_name + '/' + "rag_" + str(args.k) + '/' + args.llm_name.split('/')[-1]  + '/' + args.corpus_name  + '/' + args.retriever_name
    else:
        save_dir = args.results_dir + '/' + args.dataset_name  + '/' + "cot"  + '/' + args.llm_name.split('/')[-1]
    # import pdb; pdb.set_trace()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving to {save_dir}")

    args_dict = vars(args)
    with open(save_dir+"/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)
    # Initialize MedRAG with RAG enabled (no llm_name since it has a default)
    #model_name = "/home/zentek/DeepSeek-R1-Distill-Llama-8B"  # Generic name since llm_name is default
    # model_name = ""  # Generic name since llm_name is default

    # medrag = MedRAG(rag=True, retriever_name="RRF-3", corpus_name="Textbooks", llm_name=model_name, corpus_cache=True)
    # medrag = MedRAG(rag=rag, retriever_name="RRF-3", corpus_name="TestCorp", llm_name=model_name, HNSW=True)
    medrag = MedRAG(rag=args.rag, retriever_name=args.retriever_name, corpus_name=args.corpus_name, llm_name=args.llm_name, HNSW=args.HNSW, db_dir = "./corpus")

    # medrag = MedRAG(rag=False, retriever_name="MedCPT", corpus_name="Textbooks")
    # cot = MedRAG(llm_name="/home/zentek/Qwen2.5-7B-Instruct", rag=False)

    # MedCorp RRF-4

    # Load test data
    # test_data = []

    # questions = []
    # options = []
    # correct_answers = []



    # dataset_names = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
    # 
    # dataset_name = "mmlu"

    # dataset = QADataset(args.dataset_name)
    # for data in dataset:
    #     questions.append(data['question'])
    #     options.append(data['options'])
    #     correct_answers.append(data['answer'])


    # def batch_data(data_list, batch_size=1):
    # batch_data = []
    # for i in range(0, len(data_list), batch_size):
    #     batch_data.append(data_list[i:i + batch_size])
    # return batch_data

    with open('/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/clinical-llm-benchmark/dataset_raw/106.MIMIC-III Outcome.Diagnosis.SFT.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = []
    options = []
    correct_answers = []

    for data in dataset:
        questions.append(data['instruction'])
        options.append(data['input'])
        correct_answers.append(data['output'])


    # with open("/home/crooy/MedRAG/data/phrases_no_exclude_test.jsonl", "r", encoding="utf-8") as file:
    #     for line in file:
    #         # test_data.append(json.loads(line))
    #         data = json.loads(line)
    #         questions.append(data['question'])
    #         options.append(data['options'])
    #         correct_answers.append(data['answer_idx'])

    # import pdb; pdb.set_trace()
    # answer, _, _ = cot.answer(question=test_data[0]['question'], options=test_data[0]['options'])
    # print(f"Final answer in json with rationale: {answer}")
    # test_data=test_data[:30]
    # Initialize counters
    # total_questions = len(test_data)
    total_questions = len(questions)
    correct_predictions = 0
    error_examples = []
    # batch_size = 10

    # batched_data = batch_data(test_data, batch_size)
    batched_questions = batch_data(questions, args.batch_size)
    batched_options = batch_data(options, args.batch_size)
    batched_correct_answers = batch_data(correct_answers, args.batch_size)


    # Process each test sample
    # for idx, item in enumerate(tqdm(test_data, desc='测试进度', unit='样本'), start=1):
    save_dict = []
    for idx, (question, option, correct_answer) in enumerate(zip(batched_questions, batched_options, batched_correct_answers), start=1):
        # question = item["question"]
        # options = item["options"]
        # correct_answer = item["answer_idx"]

        # Get the answer from MedRAG
        # answer_medrag, snippets, scores = medrag.answer(question=question[0], options=option[0], k=5)
        answer_medrag, snippets, scores = medrag.batch_answer(questions=question, options=option, k=5)

        # import pdb; pdb.set_trace()
        # Parse the MedRAG output
        # Expected format: Dict{"step_by_step_thinking": "...", "answer_choice": "X"}
        
        for i, (answer) in enumerate(answer_medrag):
            save_dict.append({"question":question[i], "option":options[i], "generate_text":answer, "answer":correct_answer[i]})
                        
            # if isinstance(answer, str):
            #     match = re.search(r'"answer_choice":\s*"([A-D])"', answer, re.IGNORECASE)
            #     model_answer = match.group(1).upper() if match else ""
            # elif isinstance(answer, dict):
            #     model_answer = answer.get("answer_choice", "").upper()
            # else:
            #     model_answer = ""
            model_answer = answer.split('"answer_choice": "')[-1].strip()
            model_answer = locate_answer(model_answer)

            # Print the model's answer for debugging
            print(f"Model Answer: {model_answer}")

            # Check if the model's answer is correct
            if model_answer == correct_answer[i].upper():
                correct_predictions += 1
            else:
                error_examples.append((idx * args.batch_size + i, question, correct_answer, model_answer))
        with open(save_dir+"/model_output.json", 'w') as file:
            json.dump(save_dict, file)
        # if isinstance(answer_medrag, str):
        #     match = re.search(r'"answer_choice":\s*"([A-D])"', answer_medrag, re.IGNORECASE)
        #     model_answer = match.group(1).upper() if match else ""
        # elif isinstance(answer_medrag, dict):
        #     model_answer = answer_medrag.get("answer_choice", "").upper()
        # else:
        #     model_answer = ""

        # # Print the model's answer for debugging
        # print(f"Model Answer: {model_answer}")

        # # Check if the model's answer is correct
        # if model_answer == correct_answer.upper():
        #     correct_predictions += 1
        # else:
        #     error_examples.append((idx, question, correct_answer, model_answer))

    # Calculate accuracy
    accuracy = (correct_predictions / total_questions) * 100
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Error examples count: {len(error_examples)}")

    print(f"模型的准确率为：{accuracy:.2f}%")
    with open(save_dir+"/result.txt", 'w') as file:
        json.dump(f"Model accuracy is {accuracy:.2f}%", file)

    # # Print error examples
    # for idx, question, correct_answer, model_answer in error_examples:
    #     print(f"Question {idx} - Correct: {correct_answer} | Predicted: {model_answer}")

    # # Define output filenames
    # model_filename = os.path.basename(model_name.rstrip(os.sep))
    # accuracy_filename = f"{model_filename}_accuracy_and_model.txt"
    # error_filename = f"{model_filename}_error_questions.txt"

    # # Save accuracy to file
    # with open(accuracy_filename, "w") as accuracy_file:
    #     accuracy_file.write(f"Model: {model_name}\nAccuracy: {accuracy:.2f}%\n")

    # # Save error examples to file
    # with open(error_filename, "w") as error_file:
    #     for idx, question, correct_answer, model_answer in error_examples:
    #         error_file.write(f"Question {idx}: {question} | Correct: {correct_answer} | Predicted: {model_answer}\n")