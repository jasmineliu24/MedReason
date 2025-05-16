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
# model_name=llama70b
model_name=deepseek8b

# # Model config:
temperature=0.6
top_p=0.9
top_k=20

gpus=0,1
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
CUDA_VISIBLE_DEVICES=$gpus nohup python /n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/MedRAG/test_MedRAG_coding.py \
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
    > log/${model_name}.${inference_mode}.${now}.log 2>&1 &