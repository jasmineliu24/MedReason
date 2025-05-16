model_name=Mistral-Small-24B-Instruct-2501
# # Size: Small
# "Phi-4"
# "BioMistral-7B"
# "Ministral-8B-Instruct-2410"
# "MMed-Llama-3-8B"
# # Size: Mid
# "gemma-2-27b-it"
# "QwQ-32B-Preview"
# # Size: Large
# "Llama-3.1-70B-Instruct"
# "Llama-3.3-70B-Instruct"
# "Llama-3.1-Nemotron-70B-Instruct-HF"
# "meditron-70b"
# "MeLLaMA-70B-chat"
# "Llama3-OpenBioLLM-70B"
# "Llama-3-70B-UltraMedical"
# "Mistral-Large-Instruct-2411"
# "Phi-3.5-MoE-instruct"
# "Qwen2.5-72B-Instruct"
# "Athene-V2-Chat"
path_dir_rewrite=data/rewrite_test
gpus=2,3
# 0,1,2,3
# num_gpus=4

k=5
n=5

# get the current time: MM-DD_HH-MM
now=$(date +"%m-%d_%H-%M")

CUDA_VISIBLE_DEVICES=$gpus nohup python Calculate_ngram_vllm.py \
    --model_name $model_name \
    --path_dir_rewrite $path_dir_rewrite \
    --k $k \
    --ngram $n \
    > log/ngram_gap/${model_name}.${now}.log 2>&1 &