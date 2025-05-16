models=(
    # # Size: Small
    # "BioMistral-7B"
    # "Ministral-8B-Instruct-2410"
    # "MMed-Llama-3-8B"
    # # Size: Mid
    # "Phi-4"
    # "gemma-2-27b-it"
    # "QwQ-32B-Preview"
    # Size: Large
    "Llama-3.1-70B-Instruct"
    "Llama-3.3-70B-Instruct"
    "Llama-3.1-Nemotron-70B-Instruct-HF"
    "meditron-70b"
    "MeLLaMA-70B-chat"
    "Llama3-OpenBioLLM-70B"
    "Llama-3-70B-UltraMedical"
    "Mistral-Large-Instruct-2411"
    "Phi-3.5-MoE-instruct"
    "Qwen2.5-72B-Instruct"
    "Athene-V2-Chat"
)

split=test
batch_size=4
gpus=4,5,6,7

# get the current time: MM-DD_HH-MM
now=$(date +"%m-%d_%H-%M")

# run the model
for model_name in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpus nohup python Calculate_ppl.py \
        --model_name $model_name \
        --split $split \
        --batch_size $batch_size \
        > log/ppl/${model_name}.${now}.log 2>&1 &
done