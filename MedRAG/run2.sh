#dataset_names = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
# python test.py --llm_name /home/zentek/Qwen2.5-7B-Instruct --HNSW --k 32 --corpus_name "Textbooks" --retriever_name "MedCPT" --dataset_name 'mmlu' --batch_size 100
# python test.py --llm_name /home/zentek/Qwen2.5-7B-Instruct --HNSW --k 32 --corpus_name "Textbooks" --retriever_name "MedCPT" --dataset_name 'mmlu' --batch_size 100 --rag
# python test.py --llm_name /n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/Llama-3.3-70B-Instruct --HNSW --k 32 --corpus_name "Textbooks" --retriever_name "MedCPT" --dataset_name 'medmcqa' --batch_size 128 --rag
# python test.py --llm_name /n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/DeepSeek-R1-Distill-Llama-8B --HNSW --k 32 --corpus_name "MedCorp" --retriever_name "MedCPT" --dataset_name 'medqa' --batch_size 512 --rag

python test.py --llm_name /n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/Llama-3.3-70B-Instruct --HNSW --k 32 --corpus_name "Textbooks" --retriever_name "MedCPT" --dataset_name 'medmcqa' --batch_size 128

# export OPENAI_API_KEY=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 
# python test.py --llm_name OpenAI/gpt-3.5-turbo-16k --HNSW --k 32 --corpus_name "MedCorp" --retriever_name "MedCPT" --dataset_name 'medqa' --batch_size 1 --rag