models=(
    # "Ministral-8B-Instruct-2410"
    # "MMed-Llama-3-8B"
    # "MeLLaMA-70B-chat"
    # "Llama-3-70B-UltraMedical"
    "Qwen2.5-72B-Instruct"
    "Athene-V2-Chat"
    "Llama-3.1-Nemotron-70B-Instruct-HF"
    "Llama-3.1-70B-Instruct"
    "Mistral-Large-Instruct-2411"
    "Qwen2.5-72B-Instruct"
    "gemma-2-27b-it"
    "QwQ-32B-Preview"
    "Yi-1.5-34B-Chat-16K"
    # "Phi-4"
    # "BioMistral-7B"
    # # # Size: Small             
    # # "BioMistral-7B"
    # # "Ministral-8B-Instruct-2410"
    # # "MMed-Llama-3-8B"
    # # # Size: Mid
    # "Phi-4"
    # # "gemma-2-27b-it"
    # # "QwQ-32B-Preview"
    # # Size: Large
    # # "Llama-3.1-70B-Instruct"
    # "Llama-3.3-70B-Instruct"
    # # "Llama-3.1-Nemotron-70B-Instruct-HF"
    # "meditron-70b"
    # # "MeLLaMA-70B-chat"
    # "Llama3-OpenBioLLM-70B"
    # # "Llama-3-70B-UltraMedical"
    # # "Mistral-Large-Instruct-2411"
    # "Phi-3.5-MoE-instruct"
    # # "Qwen2.5-72B-Instruct"
    # # "Athene-V2-Chat"
    # "Yi-1.5-34B-Chat-16K"
)

path_dir_rewrite=data/rewrite_test
gpus=4,5,6,7
# 0,1,2,3

k=5
n=5

# get the current time: MM-DD_HH-MM
now=$(date +"%m-%d_%H-%M")

# run the model
for model_name in "${models[@]}"; do
    echo "Running $model_name at $now ($((++i))/${#models[@]})"
    CUDA_VISIBLE_DEVICES=$gpus nohup python Calculate_ngram_vllm.py \
    --model_name $model_name \
    --path_dir_rewrite $path_dir_rewrite \
    --k $k \
    --ngram $n \
    > log/ngram_gap/${model_name}.${now}.log 2>&1
done

# nohup bash run_ngram_mul.sh > log/ngram_gap/run_ngram_mul.log 2>&1 &
# 3936835