#dataset_names = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
# python test.py --llm_name /home/zentek/Qwen2.5-7B-Instruct --HNSW --k 32 --corpus_name "Textbooks" --retriever_name "MedCPT" --dataset_name 'mmlu' --batch_size 100
# python test.py --llm_name /home/zentek/Qwen2.5-7B-Instruct --HNSW --k 32 --corpus_name "Textbooks" --retriever_name "MedCPT" --dataset_name 'mmlu' --batch_size 100 --rag
# export VLLM_LOGGING_LEVEL=DEBUG 
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
python improved_test_script.py --llm_name "/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/DeepSeek-R1-Distill-Llama-8B" --HNSW --k 32 --corpus_name "MedCorp" --retriever_name "MedCPT" --dataset_name 'medqa' --batch_size 256 --rag