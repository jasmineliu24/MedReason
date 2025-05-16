#!/bin/bash
export TOKENIZERS_PARALLELISM=False

# Single task or multitask, which "+" join the tasks
tasks=1

# Inference mode, load the model with Huggingface or vLLM
inference_mode=vllm
# # Options:
#       vllm 
#       hf

# Support multiple models
models=(
    # Open-resource
    # "gemma-2-9b-it"
    # "gemma-2-27b-it"
    # "Llama-3.1-8B-Instruct"
    # "Llama-3.1-70B-Instruct"
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.3-70B-Instruct"
    # "Llama-3.1-Nemotron-70B-Instruct-HF"
    # "meditron-7b"
    # "meditron-70b"
    # "MeLLaMA-13B-chat"
    # "MeLLaMA-70B-chat"
    # "Llama3-OpenBioLLM-8B"
    # "Llama3-OpenBioLLM-70B"
    # "MMed-Llama-3-8B"
    # "Llama-3.1-8B-UltraMedical"
    # "Llama-3-70B-UltraMedical"
    # "Ministral-8B-Instruct-2410"
    # "Mistral-Small-Instruct-2409"
    # "Mistral-Large-Instruct-2411"
    # "BioMistral-7B"
    # "Phi-3.5-mini-instruct"
    # "Phi-3.5-MoE-instruct"
    # "Phi-4"
    # "Qwen2.5-1.5B-Instruct"
    # "Qwen2.5-3B-Instruct"
    # "Qwen2.5-7B-Instruct"
    # "Qwen2.5-72B-Instruct"
    "QwQ-32B-Preview"
    "Athene-V2-Chat"
    "Yi-1.5-9B-Chat-16K"
    "Yi-1.5-34B-Chat-16K"
    # o1-like
    # Deepseek
    # Baichun-Med-1.5B
    
    
    # Close-resource
    # "gpt-35-turbo-0125"
    # "gpt-4o-0806"
    # "o1-preview"
    # "Med-PaLM"
    # "Med-PaLM-2"
    # "Med-Gemini"
    )

# # Model config:
temperature=0.6
top_p=0.9
top_k=20

gpus=0,1,2,3
# # Options:
#       4,5,6,7
#       0,1,2,3
#       0,1,2,3,4,5,6,7

batch_size=32
max_token_input=7168
max_token_output=1024

# get the current time: MM-DD_HH-MM
now=$(date +"%m-%d_%H-%M")

# run the model
for model_name in "${models[@]}"; do
    # print the model name, the time, and the "current model index /total models"
    echo "Running $model_name at $now ($((++i))/${#models[@]})"
    CUDA_VISIBLE_DEVICES=$gpus nohup python main.py \
        --tasks $tasks \
        --inference_mode $inference_mode \
        --model_name $model_name \
        --gpus $gpus \
        --temperature $temperature \
        --top_p $top_p \
        --top_k $top_k \
        --batch_size $batch_size \
        --max_token_input $max_token_input \
        --max_token_output $max_token_output \
        > log/${model_name}.${inference_mode}.${now}.log 2>&1 
done

# nohup bash run_multiple_model.sh > log/run_multiple_model.log 2>&1 &
# pid: 3200707

# nohup bash run_multiple_model.sh > log/run_multiple_model_2.log 2>&1 &
# pid: 3201302