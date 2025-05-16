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
        "--batch_size",
        type=int,
        default=8,
        help="The batch_size for the model.",
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

    # exp config
    model_type = "chat"

    # load model
    model, tokenizer, max_token_all = load_model(args.model_name)

    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    # input and output path
    dict_task_data = list_benbench_tasks(path=args.path_dir_rewrite)
    print("The number of tasks: ", len(dict_task_data))

    dict_task_data = {
        key: value
        for key, value in dict_task_data.items()
        if key
        in [
            # # 0
            # "1-1.ADE-ADE identification.SFT|test",
            # "1-2.ADE-ADE relation.SFT|test",
            # "1-3.ADE-Drug dosage.SFT|test",
            # "100.GraSSCo_PHI.SFT|test",
            # "101.IFMIR.IncidentType.SFT|test",
            # "101.IFMIR.NER.SFT|test",
            # "101.IFMIR.NER_factuality.SFT|test",
            # "102.iCorpus.SFT|test",
            # "103.icliniq-10k.SFT|test",
            # "104.HealthCareMagic-100k.SFT|test",
            # # 1
            # "105.MIMIC-IV CDM.SFT|test",
            # "106.MIMIC-III Outcome.LoS.SFT|test",
            # "106.MIMIC-III Outcome.Mortality.SFT|test",
            # "107.MIMIC-IV BHC.SFT|test",
            # "108.MIMIC-IV DiReCT.Dis.SFT|test",
            # "108.MIMIC-IV DiReCT.PDD.SFT|test",
            # "12.C-EMRS.SFT|test",
            # "17-1.CLEF_eHealth_2020_CodiEsp_corpus-ICD-10-CM.SFT|test",
            # "17-2.CLEF_eHealth_2020_CodiEsp_corpus-ICD-10-PCS.SFT|test",
            # "19.ClinicalNotes-UPMC.SFT|test",
            # 2
            # "20.clinical records from the Mexican Social Security Institute.SFT|test",
            # "21.CLINpt.SFT|test",
            # "22.CLIP.SFT|test",
            # "23.cMedQA.SFT|test",
            # "26.DialMed.SFT|test",
            # "27.DiSMed.SFT|test",
            # "28.MIE.SFT|test",
            # "29.EHRQA.primary_department.SFT|test",
            # "29.EHRQA.qa.SFT|test",
            # "29.EHRQA.sub_department.SFT|test",
            # # 3
            # "3-2.BARR2-resolution.SFT|test",
            # "31.Ex4CDS.SFT|test",
            # "33.GOUT-CC.consensus.SFT|test",
            # "33.GOUT-CC.predict.SFT|test",
            # "35.n2c2 2006 - De-identification.SFT|test",
            # "37.i2b2-2009-Medication-Extraction-Challenge.SFT|test",
            # "38-1.i2b2-2010-Relations-Challenge-concept.SFT|test",
            # "38-2.i2b2-2010-Relations-Challenge-assertion.SFT|test",
            # "38-3.i2b2-2010-Relations-Challenge-relation.SFT|test",
            # "41.n2c2 2014 - De-identification.SFT|test",
            # # 4
            # "43.IMCS-V2-NER.SFT|test",
            # "46.Japanese Case Reports.SFT|test",
            # "48.meddocan.SFT|test",
            # "5.BrainMRI-AIS.SFT|test",
            # "51.MEDIQA_2019_Task2_RQE.SFT|test",
            # "55.MedNLI.SFT|test",
            # "57.MedSTS.SFT|test",
            # "6.Brateca.hospitalization.SFT|test",
            # "6.Brateca.mortality.SFT|test",
            # "62.mtsamples.SFT|test",
            # # 5
            # "63.MTSamples-temporal annotation.SFT|test",
            # "65.n2c2-2018-Track2-Adverse-Drug-Events-and-Medication-Extraction.SFT|test",
            # "66-1.NorSynthClinical-entity.SFT|test",
            # "66-2.NorSynthClinical-relation.SFT|test",
            # "68.NUBES.SFT|test",
            # "7.Cantemist.CODING.SFT|test",
            # "7.Cantemist.NER.SFT|test",
            # "7.Cantemist.Norm.SFT|test",
            # "76-1.MTS-Dialog-MEDIQA-2023-chat-task-A.SFT|test",
            # "76-2.MTS-Dialog-MEDIQA-2023-sum-task-A.SFT|test",
            # # 6
            # "76-3.MTS-Dialog-MEDIQA-2023-sum-task-B.SFT|test",
            # "8.CARES.area.SFT|test",
            # "8.CARES.icd10_block.SFT|test",
            # "8.CARES.icd10_chapter.SFT|test",
            # "8.CARES.icd10_sub_block.SFT|test",
            # "80.RuMedDaNet.SFT|test",
            # "81.CHIP-CDN.SFT|test",
            # "82.CHIP-CTC.SFT|test",
            # "83.CHIP-MDCFNPC.SFT|test",
            # "84.MedDG.SFT|test",
            # # 7
            # "85.IMCS-V2-SR.SFT|test",
            # "86.IMCS-V2-MRG.SFT|test",
            # "87.IMCS-V2-DAC.SFT|test",
            # "9.CHIP-CDEE.SFT|test",
            # "90-1.n2c2 2014 - Heart Disease Challenge - Diabete.SFT|test",
            # "90-2.n2c2 2014 - Heart Disease Challenge - CAD.SFT|test",
            # "90-3.n2c2 2014 - Heart Disease Challenge - Hyperlipidemia.SFT|test",
            # "90-4.n2c2 2014 - Heart Disease Challenge - Hypertension.SFT|test",
            # "90-8.n2c2 2014 - Heart Disease Challenge - Medication.SFT|test",
            # "91-1.CAS.label.SFT|test",
            # # 8
            # "91-2.CAS.evidence.SFT|test",
            # "93.RuMedNLI.SFT|test",
            # "94.RuDReC.SFT|test",
            # "95.NorSynthClinical-PHI.SFT|test",
            # "96.RuCCoN.NER.SFT|test",
            # "97.CLISTER.SFT|test",
            # "98.BRONCO150.NER_status.SFT|test",
            # "99.CARDIO_DE.SFT|test",
        ]
    }
    print(" - After filtering: ", len(dict_task_data))

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

        path_file_orgn = os.path.join(task_path, f"ngrams-{args.model_name}-orgn.jsonl")
        ngrams_result_orgn = calculate_my_n_gram_accuracy_hf_batch(
            n=args.ngram,
            k=args.k,
            dataset=orgn_data,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            path_file_output=path_file_orgn,
            batch_size=args.batch_size,
            model_type=model_type,
        )
        print(f"orgn {args.ngram}-gram_acc_avg: ", ngrams_result_orgn["mean_n_grams"])
        ngram_list.append(ngrams_result_orgn["mean_n_grams"])

        # path_file_rewritten1 = os.path.join(
        #     task_path, f"ngrams-{args.model_name}-rewritten1.jsonl"
        # )
        # ngrams_result_rewritten1 = calculate_my_n_gram_accuracy_hf_batch(
        #     n=args.ngram,
        #     k=args.k,
        #     dataset=rewritten_data1,
        #     model=model,
        #     tokenizer=tokenizer,
        #     model_name=args.model_name,
        #     path_file_output=path_file_rewritten1,
        #     batch_size=args.batch_size,
        #     model_type=model_type,
        # )
        # print(
        #     f"re-1 {args.ngram}-gram_acc_avg: ",
        #     ngrams_result_rewritten1["mean_n_grams"],
        # )
        # ngram_list.append(ngrams_result_rewritten1["mean_n_grams"])

        # path_file_rewritten2 = os.path.join(
        #     task_path, f"ngrams-{args.model_name}-rewritten2.jsonl"
        # )
        # ngrams_result_rewritten2 = calculate_my_n_gram_accuracy_hf_batch(
        #     n=args.ngram,
        #     k=args.k,
        #     dataset=rewritten_data2,
        #     model=model,
        #     tokenizer=tokenizer,
        #     model_name=args.model_name,
        #     path_file_output=path_file_rewritten2,
        #     batch_size=args.batch_size,
        #     model_type=model_type,
        # )
        # print(
        #     f"re-2 {args.ngram}-gram_acc_avg: ",
        #     ngrams_result_rewritten2["mean_n_grams"],
        # )
        # ngram_list.append(ngrams_result_rewritten2["mean_n_grams"])

        # path_file_rewritten3 = os.path.join(
        #     task_path, f"ngrams-{args.model_name}-rewritten3.jsonl"
        # )
        # ngrams_result_rewritten3 = calculate_my_n_gram_accuracy_hf_batch(
        #     n=args.ngram,
        #     k=args.k,
        #     dataset=rewritten_data3,
        #     model=model,
        #     tokenizer=tokenizer,
        #     model_name=args.model_name,
        #     path_file_output=path_file_rewritten3,
        #     batch_size=args.batch_size,
        #     model_type=model_type,
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
