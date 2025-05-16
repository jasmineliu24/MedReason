import os
import re
import json
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

class MedRAG:
    def __init__(self, llm_name="DeepSeek-R1-Distill-Llama-8B", rag=True, retriever_name="MedCPT", 
                 corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, HNSW=False):
        self.llm_name = llm_name
        print(f"Using vLLM for inference with model: {self.llm_name}")
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
       
        # Import retrieval system only if RAG is enabled
        if rag:
            from src.utils import RetrievalSystem
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=False, HNSW=HNSW)
        else:
            self.retrieval_system = None
        
        # Templates for ICD-9 code prediction
        self.templates = {
            "cot_system": "You are a helpful medical expert, and your task is to predict 3-digit integer ICD-9 clinical codes. Please first think step-by-step and then output the answers. Organize your output in a json formatted as Dict{\"step_by_step_thinking\": Str(explanation), \"answer_choice\": Str{ICD-9 code(s)}}. Your responses will be used for research purposes only, so please have a definite answer.",
            "cot_prompt": "Here is the patient information:\n{question}\n\n{options}\n\nPlease think step-by-step and generate your output in json:",
            "medrag_system": "You are a helpful medical expert, and your task is to predict 3-digit integer ICD-9 clinical codes using the relevant documents and patient information. Please first think step-by-step and then output the answers. Organize your output in a json formatted as Dict{\"step_by_step_thinking\": Str(explanation), \"answer_choice\": Str{ICD-9 code(s)}}. Your responses will be used for research purposes only, so please have a definite answer.",
            "medrag_prompt": "Here are the relevant documents:\n{context}\n\nHere is the patient information:\n{question}\n\n{options}\n\nPlease think step-by-step and generate your output in json:"
        }
        
        # Initialize tokenizer and model
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_name)
        # self.tokenizer = AutoTokenizer.from_pretrained("/n/holylfs06/LABS/kempner_undergrads/Lab/jasmineliu/Llama-3.3-8B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
        self.max_length = 2048
        self.context_length = 1024
        
        # Handle specific model configurations
        if "llama-2" in llm_name.lower():
            self.max_length = 4096
            self.context_length = 3072
        elif "llama-3" in llm_name.lower():
            self.max_length = 8192
            self.context_length = 7168
        
        # Initialize vLLM model
        print("Loading LLM...")
        self.model = LLM(
            model=self.llm_name,
            # tensor_parallel_size=4,  # Adjust based on your GPU setup
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams(temperature=0.0, max_tokens=4096)
        print("LLM loading completed")

    def generate(self, messages):
        """Generate model response for given messages"""
        response = self.model.generate(messages, self.SamplingParams)
        result_texts = []
        for output in response:
            result_texts.append(output.outputs[0].text)
        return result_texts

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir=None):
        """
        Process a single question with RAG if enabled
        
        Args:
            question: The instruction/question
            options: The patient information
            k: Number of snippets to retrieve
            rrf_k: Parameter for Reciprocal Rank Fusion
            save_dir: Directory to save results
        """
        # Ensure options is a string
        if options is None:
            options = ''
            
        # Retrieve relevant snippets if RAG is enabled
        if self.rag:
            retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
            
            # Format contexts
            contexts = [f"Document [{idx}] (Title: {snippet['title']}) {snippet['content']}" 
                       for idx, snippet in enumerate(retrieved_snippets)]
            
            if len(contexts) == 0:
                contexts = [""]
                
            # Truncate context if necessary
            context_text = "\n".join(contexts)
            encoded_context = self.tokenizer.encode(context_text, add_special_tokens=False)
            if len(encoded_context) > self.context_length:
                context_text = self.tokenizer.decode(encoded_context[:self.context_length])
        else:
            retrieved_snippets = []
            scores = []
            context_text = ""

        # Create directory if save_dir is specified
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate answers
        if not self.rag:
            prompt = self.templates["cot_prompt"].format(question=question, options=options)
            system_prompt = self.templates["cot_system"]
            messages = [f"{system_prompt}\n\n{prompt}"]
        else:
            prompt = self.templates["medrag_prompt"].format(context=context_text, question=question, options=options)
            system_prompt = self.templates["medrag_system"]
            messages = [f"{system_prompt}\n\n{prompt}"]

        # Generate response
        answers = self.generate(messages)
        cleaned_answer = re.sub(r"\s+", " ", answers[0])
        
        # Save results if save_dir is specified
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return cleaned_answer, retrieved_snippets, scores

    def batch_answer(self, questions, options=None, k=32, rrf_k=100, save_dir=None):
        """
        Process a batch of questions with RAG if enabled
        
        Args:
            questions: List of instructions/questions
            options: List of patient information
            k: Number of snippets to retrieve
            rrf_k: Parameter for Reciprocal Rank Fusion
            save_dir: Directory to save results
        """
        batch_messages = []
        answers = []
        all_snippets = []
        all_scores = []
        
        # Process each question in the batch
        for question, option in zip(questions, options):
            # Retrieve relevant snippets if RAG is enabled
            if self.rag:
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
                
                # Format contexts
                contexts = [f"Document [{idx}] (Title: {snippet['title']}) {snippet['content']}" 
                           for idx, snippet in enumerate(retrieved_snippets)]
                
                if len(contexts) == 0:
                    contexts = [""]
                    
                # Truncate context if necessary
                context_text = "\n".join(contexts)
                encoded_context = self.tokenizer.encode(context_text, add_special_tokens=False)
                if len(encoded_context) > self.context_length:
                    context_text = self.tokenizer.decode(encoded_context[:self.context_length])
                
                all_snippets.append(retrieved_snippets)
                all_scores.append(scores)
                
                # Create prompt with context
                prompt = self.templates["medrag_prompt"].format(context=context_text, question=question, options=option)
                system_prompt = self.templates["medrag_system"]
                batch_messages.append(f"{system_prompt}\n\n{prompt}")
            else:
                all_snippets.append([])
                all_scores.append([])
                
                # Create prompt without context
                prompt = self.templates["cot_prompt"].format(question=question, options=option)
                system_prompt = self.templates["cot_system"]
                batch_messages.append(f"{system_prompt}\n\n{prompt}")
        
        # Generate responses for all messages
        responses = self.generate(batch_messages)
        
        # Clean responses
        for response in responses:
            answers.append(re.sub(r"\s+", " ", response))
        
        # Save results if save_dir is specified
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(all_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers, all_snippets, all_scores