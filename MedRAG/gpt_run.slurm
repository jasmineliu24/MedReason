#!/bin/bash
#SBATCH --job-name=llama_test    # Job name
#SBATCH --partition=kempner_h100 # GPU partition
#SBATCH --account=kempner_undergrads # Account
#SBATCH --time=04:00:00         # Time limit (4 hours)
#SBATCH --mem=320G              # Total memory
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16      # Request 16 CPU cores
#SBATCH --output=output.log      # Standard output log
#SBATCH --error=error.log        # Error log

# Load necessary modules
module load python

# Activate virtual environment (if needed)
conda activate kure_new  # Replace with your conda/venv

# Run script using torchrun for multi-GPU execution
torchrun --nproc_per_node=4 test.py \
    --llm_name /n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/DeepSeek-R1-Distill-Llama-8B \
    --HNSW --k 32 --corpus_name "PubMed" \
    --retriever_name "MedCPT" --dataset_name 'medmcqa' \
    --batch_size 128 --rag
