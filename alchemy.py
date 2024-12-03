from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
from transformers import pipeline
import os
import requests, time

def crawl_github_repo(url,is_sub_dir):

    ignore_list = ['__init__.py']

    if not is_sub_dir:
        api_url = f"https://api.github.com/repos/{url}/contents"
    else:
        api_url = url



    response = requests.get(api_url)
    response.raise_for_status()  # Check for any request errors

    files = []

    contents = response.json()

    for item in contents:
        if item['type'] == 'file' and item['name'] not in ignore_list and (item['name'].endswith('.py') or item['name'].endswith('.ipynb')):
            files.append(item['html_url'])
        elif item['type'] == 'dir' and not item['name'].startswith("."):
            sub_files = crawl_github_repo(item['url'],True)
            time.sleep(.1)
            files.extend(sub_files)

    return files

def extract_python_code_from_py(github_url):
    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    response = requests.get(raw_url)
    response.raise_for_status()  # Check for any request errors

    python_code = response.text

    return python_code



class AutoRAG:
    """
    A class for Retrieval-Augmented Generation using a specified model and dataset.
    """

    def __init__(self, model_name, dataset, embedding_model='all-MiniLM-L6-v2', **kwargs): 
        self.model_name = model_name
        self.dataset = load_dataset(dataset)
        self.pipeline = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.knowledge_base = {}  # To store chunks and their embeddings

    def __repr__(self): 
        return f"{_ for _ in self.__dict__.keys()}"

    
    





    def chunk_github_repo(self, GITHUB_REPO, chunk_size=200):
        """
        Clone and chunk the content of a GitHub repository.

        Args:
            repo_url (str): The URL of the GitHub repository.
            chunk_size (int): Number of words per chunk.


        """
        GITHUB_TOKEN = "github_token_here"
        # Example for downloading raw files (you can expand this to clone a full repo)
        code_files_urls = crawl_github_repo(GITHUB_REPO,False)

        chunks = []

        for i in range(0, len (code_files_urls)):
            if code_files_urls[i].endswith(".py"):
                content = extract_python_code_from_py(code_files_urls[i])
                #doc = Document(page_content=content, metadata= {"url": code_files_urls[i], "file_index":i})
                chunks.append(content)
        

        # Store chunks and their embeddings
        for chunk in chunks:
            embedding = self.embedding_model.encode(chunk, convert_to_tensor=True)
            self.knowledge_base[chunk] = embedding

    def retrieve_top_k_chunks(self, query, k=1):
        """
        Retrieve the top-k relevant chunks for a given query.

        Args:
            query (str): The query string.
            k (int): Number of top chunks to retrieve.

        Returns:
            List[str]: The top-k relevant chunks.
        """
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        scores = {
            chunk: util.pytorch_cos_sim(query_embedding, emb).item()
            for chunk, emb in self.knowledge_base.items()
        }
        # Sort chunks by similarity score in descending order
        top_chunks = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        return top_chunks

    def _evaluate_problem(self, problem: dict, k: int = 1) -> list[bool]:
        """
        Evaluate a single problem in the dataset with RAG.

        Args:
            problem (dict): A single problem from the dataset.
            k (int): Number of completions to generate.

        Returns:
            List[bool]: A list of booleans indicating if each generated completion passed the test.
        """
        prompt = problem["prompt"]
        test_code = problem["test"]

        # Retrieve top-k relevant chunks
        retrieved_chunks = self.retrieve_top_k_chunks(prompt, k=k)
        context = "\n".join(retrieved_chunks)

        # Augment the prompt with retrieved context
        augmented_prompt = f"{context}\n{prompt}"
        print(augmented_prompt)

        # Generate k completions
        completions = self.pipeline(augmented_prompt, num_return_sequences=k, max_new_tokens=200, truncation=True, temperature=0.7)

        results = []
        for completion in completions:
            code = completion["generated_text"]
            try:
                # Combine prompt and generated code
                exec_code = augmented_prompt + code
                
                # Execute the combined code
                local_env = {}
                exec(exec_code, {"__builtins__": None}, local_env)
                
                # Run the provided tests
                exec(test_code, {"__builtins__": None}, local_env)
                
                results.append(True)
            except Exception:
                results.append(False)
        
        return results

    def benchmark(self):
        """
        Benchmark the model on the dataset using pass@k metric.
        """
        repo_url = 'akantuni/Kattis'
        if repo_url:
            print(f"Processing GitHub repository from {repo_url}...")
            self.chunk_github_repo(repo_url)
            print(f"Repository content stored in knowledge base. Total chunks: {len(self.knowledge_base)}")
        k = 1
        all_results = []
        for problem in self.dataset["test"]:
            results = self._evaluate_problem(problem, k=k)
            all_results.append(results)

        pass_at_k = self._calculate_pass_at_k(all_results, k=k)
        print(f"Pass@{k}: {pass_at_k:.4f}")

    @staticmethod
    def _calculate_pass_at_k(results: list[list[bool]], k: int) -> float:
        """
        Calculate pass@k for a list of results.

        Args:
            results (List[List[bool]]): Results for each prompt, each containing a list of booleans.
            k (int): The number of samples considered.

        Returns:
            float: The pass@k score.
        """
        num_problems = len(results)
        pass_count = 0
        
        for problem_results in results:
            pass_count += any(problem_results[:k])
        
        return pass_count / num_problems

    
if __name__ == "__main__":

    auto = AutoRAG(
        model_name = "EleutherAI/gpt-neo-1.3B",
        dataset = "openai_humaneval",
    )

    print(auto)

    auto.benchmark()





