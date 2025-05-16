from model.init import seed_everything
from metric.classification import *
from metric.extraction import *
from metric.generation import *
from dataset.classification import *
from dataset.extraction import *
from dataset.generation import *
from dataset.config import *


class EmptyArgs:
    def __init__(self):
        pass


args = EmptyArgs()

num_seed = 42
seed_everything(seed=num_seed)

# Configuration
num_bootstrap = 1000
path_dir_performance = "performance"
list_prompt_mode = ["direct", "cot", "direct-5-shot"]
model_to_evaluate = [
    # "Baichuan-M1-14B-Instruct",
    # "DeepSeek-R1",
    # "DeepSeek-R1-Distill-Llama-8B",
    # "DeepSeek-R1-Distill-Llama-70B",
    # "DeepSeek-R1-Distill-Qwen-1.5B",
    # "DeepSeek-R1-Distill-Qwen-7B",
    # "DeepSeek-R1-Distill-Qwen-14B",
    # "DeepSeek-R1-Distill-Qwen-32B",
    # "gemma-2-9b-it",
    # "gemma-2-27b-it",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    # "Llama-3.1-8B-Instruct",
    # "Llama-3.1-70B-Instruct",
    # "Llama-3.2-1B-Instruct",
    # "Llama-3.2-3B-Instruct",
    # "Llama-3.3-70B-Instruct",
    "Llama-4-Scout-17B-16E-Instruct",
    # "Llama-3.1-Nemotron-70B-Instruct-HF",
    # "meditron-7b",
    # "meditron-70b",
    # "MeLLaMA-13B-chat",
    # "MeLLaMA-70B-chat",
    # "Llama3-OpenBioLLM-8B",
    # "Llama3-OpenBioLLM-70B",
    # "MMed-Llama-3-8B",
    # "Llama-3.1-8B-UltraMedical",
    # "Llama-3-70B-UltraMedical",
    # "Ministral-8B-Instruct-2410",
    # "Mistral-Small-Instruct-2409",
    # "Mistral-Small-24B-Instruct-2501",
    "Mistral-Small-3.1-24B-Instruct-2503",
    # "Mistral-Large-Instruct-2411",
    # "BioMistral-7B",
    # "Phi-3.5-mini-instruct",
    # "Phi-3.5-MoE-instruct",
    # "Phi-4",
    # "Qwen2.5-1.5B-Instruct",
    # "Qwen2.5-3B-Instruct",
    # "Qwen2.5-7B-Instruct",
    # "Qwen2.5-72B-Instruct",
    # "QwQ-32B-Preview",
    # "QWQ-32B",
    # "Athene-V2-Chat",
    # "Yi-1.5-9B-Chat-16K",
    # "Yi-1.5-34B-Chat-16K",
    # "gpt-35-turbo",
    # "gpt-4o",
    # "gemini-2.0-flash",
    # "gemini-1.5-pro",
]

model_to_print = get_models_evaluate()

# Classfication
all_tasks_clf = {
    "1-1.ADE-ADE identification": Task_clf_ADE_ADE_identification,
    "5.BrainMRI-AIS": Task_clf_Brain_MRI_AIS,
    "6.Brateca.hospitalization": Task_clf_Brateca_hospitalization,
    "6.Brateca.mortality": Task_clf_Brateca_mortality,
    "7.Cantemist.CODING": Task_nor_Cantemist_CODING,
    "8.CARES.area": Task_clf_CARES_area,
    "8.CARES.icd10_chapter": Task_clf_CARES_icd10_chapter,
    "8.CARES.icd10_block": Task_nor_CARES_icd10_block,
    "8.CARES.icd10_sub_block": Task_nor_CARES_icd10_sub_block,
    "12.C-EMRS": Task_clf_C_EMRS,
    "19.ClinicalNotes-UPMC": Task_clf_Clinical_Notes_UPMC,
    "20.clinical records from the Mexican Social Security Institute": Task_clf_clinical_records_from_the_Mexican_Social_Security_Institute,
    "22.CLIP": Task_clf_CLIP,
    "26.DialMed": Task_clf_Dial_Med,
    "29.EHRQA.primary_department": Task_clf_EHRQA_primary_department,
    "29.EHRQA.sub_department": Task_clf_EHRQA_sub_department,
    "33.GOUT-CC.consensus": Task_clf_GOUT_CC_consensus,
    "46.Japanese Case Reports": Task_clf_Japanese_Case_Reports,
    "51.MEDIQA_2019_Task2_RQE": Task_clf_MEDIQA_2019_Task2_RQE,
    "55.MedNLI": Task_clf_MedNLI,
    "57.MedSTS": Task_clf_MedSTS,
    "62.mtsamples": Task_clf_mtsamples,
    "76-2.MTS-Dialog-MEDIQA-2023-sum-task-A": Task_clf_MTS_Dialog_MEDIQA_2023_sum_task_A,
    "80.RuMedDaNet": Task_clf_RuMedDaNet,
    "81.CHIP-CDN": Task_nor_CHIP_CDN,
    "82.CHIP-CTC": Task_clf_CHIP_CTC,
    "87.IMCS-V2-DAC": Task_clf_IMCS_V2_DAC,
    "93.RuMedNLI": Task_clf_RuMedNLI,
    "97.CLISTER": Task_clf_CLISTER,
    "101.IFMIR.IncidentType": Task_clf_IFMIR_IncidentType,
    "105.MIMIC-IV CDM": Task_clf_mimic_iv_CDM,
    "106.MIMIC-III Outcome.LoS": Task_clf_mimic_iii_outcome_LoS,
    "106.MIMIC-III Outcome.Mortality": Task_clf_mimic_iii_outcome_Mortality,
    "108.MIMIC-IV DiReCT.Dis": Task_clf_mimic_iv_DiReCT_Dis,
    "108.MIMIC-IV DiReCT.PDD": Task_clf_mimic_iv_DiReCT_PDD,
}

# Extraction
all_tasks_ext = {
    "1-2.ADE-ADE relation": Task_ext_ADE_ADE_relation,
    "1-3.ADE-Drug dosage": Task_ext_ADE_Drug_dosage,
    "3-2.BARR2-resolution": Task_ext_BARR2_resolution,
    "7.Cantemist.NER": Task_ext_Cantemist_NER,
    "7.Cantemist.Norm": Task_ext_Cantemist_Norm,
    "9.CHIP-CDEE": Task_ext_CHIP_CDEE,
    "17-1.CLEF_eHealth_2020_CodiEsp_corpus-ICD-10-CM": Task_ext_CLEF_ICD_10_CM,
    "17-2.CLEF_eHealth_2020_CodiEsp_corpus-ICD-10-PCS": Task_ext_CLEF_ICD_10_PCS,
    "21.CLINpt": Task_ext_CLINpt,
    "27.DiSMed": Task_ext_DiSMed,
    "28.MIE": Task_ext_MIE,
    "31.Ex4CDS": Task_ext_Ex4CDS,
    "35.n2c2 2006 - De-identification": Task_ext_n2c2_2006_De_Identification,
    "37.i2b2-2009-Medication-Extraction-Challenge": Task_ext_i2b2_2009_Medication_Extraction_Challenge,
    "38-1.i2b2-2010-Relations-Challenge-concept": Task_ext_i2b2_2010_Relations_Challenge_concept,
    "38-2.i2b2-2010-Relations-Challenge-assertion": Task_ext_i2b2_2010_Relations_Challenge_assertion,
    "38-3.i2b2-2010-Relations-Challenge-relation": Task_ext_i2b2_2010_Relations_Challenge_relation,
    "41.n2c2 2014 - De-identification": Task_ext_n2c2_2014_De_identification,
    "43.IMCS-V2-NER": Task_ext_IMCS_V2_NER,
    "48.meddocan": Task_ext_meddocan,
    "63.MTSamples-temporal annotation": Task_ext_MTSamples_temporal_annotation,
    "65.n2c2-2018-Track2-Adverse-Drug-Events-and-Medication-Extraction": Task_ext_n2c2_2018_Track2_Adverse_Drug_Events_and_Medication_Extraction,
    "66-1.NorSynthClinical-entity": Task_ext_NorSynthClinical_entity,
    "66-2.NorSynthClinical-relation": Task_ext_NorSynthClinical_relation,
    "68.NUBES": Task_ext_NUBES,
    "83.CHIP-MDCFNPC": Task_ext_CHIP_MDCFNPC,
    "85.IMCS-V2-SR": Task_ext_IMCS_V2_SR,
    "90-1.n2c2 2014 - Heart Disease Challenge - Diabete": Task_ext_n2c2_2014_Heart_Disease_Challenge_Diabete,
    "90-2.n2c2 2014 - Heart Disease Challenge - CAD": Task_ext_n2c2_2014_Heart_Disease_Challenge_CAD,
    "90-3.n2c2 2014 - Heart Disease Challenge - Hyperlipidemia": Task_ext_n2c2_2014_Heart_Disease_Challenge_Hyperlipidemia,
    "90-4.n2c2 2014 - Heart Disease Challenge - Hypertension": Task_ext_n2c2_2014_Heart_Disease_Challenge_Hypertension,
    "90-8.n2c2 2014 - Heart Disease Challenge - Medication": Task_ext_n2c2_2014_Heart_Disease_Challenge_Medication,
    "91-1.CAS.label": Task_ext_CAS_label,
    "94.RuDReC": Task_ext_RuDReC,
    "95.NorSynthClinical-PHI": Task_ext_NorSynthClinical_PHI,
    "96.RuCCoN.NER": Task_ext_RuCCoN_NER,
    "98.BRONCO150.NER_status": Task_ext_BRONCO150_NER_status,
    "99.CARDIO:DE": Task_ext_CARDIO_DE,
    "100.GraSSCo_PHI": Task_ext_GraSSCo_PHI,
    "101.IFMIR.NER": Task_ext_IFMIR_NER,
    "101.IFMIR.NER_factuality": Task_ext_IFMIR_NER_factuality,
    "102.iCorpus": Task_ext_iCorpus,
}

# Generation
all_tasks_gen = {
    "23.cMedQA": Task_gen_cMedQA,
    "29.EHRQA.qa": Task_gen_EHRQA_qa,
    "76-1.MTS-Dialog-MEDIQA-2023-chat-task-A": Task_gen_MTS_Dialog_MEDIQA_2023_chat_task_A,
    "76-3.MTS-Dialog-MEDIQA-2023-sum-task-B": Task_gen_MTS_Dialog_MEDIQA_2023_sum_task_B,
    "84.MedDG": Task_gen_MedDG,
    "86.IMCS-V2-MRG": Task_gen_IMCS_V2_MRG,
    "91-2.CAS.evidence": Task_gen_CAS_evidence,
    "103.icliniq-10k": Task_gen_icliniq,
    "104.HealthCareMagic-100k": Task_gen_HealthCareMagic,
    "107.MIMIC-IV BHC": Task_gen_mimic_iv_BHC,
}


def evaluate(task):
    dict_prompt_model_performance = {}
    for prompt_mode in list_prompt_mode:
        dict_model_performance = task.evaluate_by_model(
            prompt_mode=prompt_mode,
            model_name=model_to_evaluate,
            bootstrap=num_bootstrap,
        )
        path_file_performance = (
            f"{path_dir_performance}/{task.name}.{prompt_mode}.performance.json"
        )
        # If the file exists, load the existing file and update the performance. Otherwise, create a new file.
        if os.path.exists(path_file_performance):
            with open(path_file_performance, "r") as f:
                dict_model_performance_old = json.load(f)
            for model_name in dict_model_performance.keys():
                dict_model_performance_old[model_name] = dict_model_performance[
                    model_name
                ]
            dict_model_performance = dict_model_performance_old
            with open(path_file_performance, "w") as f:
                json.dump(dict_model_performance, f, indent=4)
        else:
            with open(path_file_performance, "w") as f:
                json.dump(dict_model_performance, f, indent=4)
        dict_prompt_model_performance[prompt_mode] = dict_model_performance
    return dict_prompt_model_performance


def print_performance(dict_prompt_model_performance, task_type):
    dict_mode_performance = {}
    for prompt_mode in list_prompt_mode:
        print("Prompt Mode:", prompt_mode)
        if task_type == "clf":
            str_metrics = print_metrics_clf(
                dict_prompt_model_performance[prompt_mode], flag_print_missing=False
            )
        elif task_type == "ext":
            str_metrics = print_metrics_ext(
                dict_prompt_model_performance[prompt_mode], flag_print_missing=False
            )
        elif task_type == "gen":
            str_metrics = print_metrics_gen(
                dict_prompt_model_performance[prompt_mode], flag_print_missing=False
            )
        else:
            raise ValueError("Invalid task type")
        print(str_metrics)
        print("===============================")
        dict_mode_performance[prompt_mode] = str_metrics
    return dict_mode_performance


def print_all_performance():
    # Classification
    print("Classification")
    for prompt_mode in list_prompt_mode:
        print(f"Prompt Mode: {prompt_mode}")
        for task_name in all_tasks_clf.keys():
            path_file_performance = (
                f"{path_dir_performance}/{task_name}.{prompt_mode}.performance.json"
            )
            with open(path_file_performance, "r") as f:
                dict_model_performance = json.load(f)
            str_metrics = print_metrics_clf(
                dict_model_performance, model_to_print, flag_print_missing=False
            )
            print(str_metrics)
        print("===============================")
    print()

    # Extraction
    print("Extraction")
    for prompt_mode in list_prompt_mode:
        print(f"Prompt Mode: {prompt_mode}")
        for task_name in all_tasks_ext.keys():
            path_file_performance = (
                f"{path_dir_performance}/{task_name}.{prompt_mode}.performance.json"
            )
            with open(path_file_performance, "r") as f:
                dict_model_performance = json.load(f)
            str_metrics = print_metrics_ext(
                dict_model_performance, model_to_print, flag_print_missing=False
            )
            print(str_metrics)
        print("===============================")
    print()

    # Generation
    print("Generation")
    for prompt_mode in list_prompt_mode:
        print(f"Prompt Mode: {prompt_mode}")
        for task_name in all_tasks_gen.keys():
            path_file_performance = (
                f"{path_dir_performance}/{task_name}.{prompt_mode}.performance.json"
            )
            with open(path_file_performance, "r") as f:
                dict_model_performance = json.load(f)
            str_metrics = print_metrics_gen(
                dict_model_performance, model_to_print, flag_print_missing=False
            )
            print(str_metrics)
        print("===============================")
    print()


def evaluate_all():
    for task, evaluation_function in all_tasks_clf.items():
        print(task)
        task = evaluation_function(args=args, task=task)
        dict_prompt_model_performance = evaluate(task)
        dict_mode_performance = print_performance(dict_prompt_model_performance, "clf")
        print()
    for task, evaluation_function in all_tasks_ext.items():
        print(task)
        task = evaluation_function(args=args, task=task)
        dict_prompt_model_performance = evaluate(task)
        dict_mode_performance = print_performance(dict_prompt_model_performance, "ext")
        print()
    for task, evaluation_function in all_tasks_gen.items():
        print(task)
        task = evaluation_function(args=args, task=task)
        dict_prompt_model_performance = evaluate(task)
        dict_mode_performance = print_performance(dict_prompt_model_performance, "gen")
        print()


if __name__ == "__main__":
    evaluate_all()
    print_all_performance()
