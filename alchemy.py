from sentence_transformers import SentenceTransformer, util
import torch
from datasets import load_dataset
from transformers import pipeline, AutoModel, AutoTokenizer
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
            time.sleep(6)
            files.extend(sub_files)
    print(files)

    return files

def crawl_multiple_repos(urls):
    all_files = []
    for url in urls:
        try:
            print(f"Crawling repo: {url}")
            repo_files = crawl_github_repo(url, False)
            all_files.extend(repo_files)
        except Exception as e:
            print(f"Error while crawling {url}: {e}")
    return all_files

def extract_python_code_from_py(github_url):
    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    response = requests.get(raw_url)
    response.raise_for_status()  # Check for any request errors

    python_code = response.text

    return python_code


class Retriever:
    def __init__(self, chunking='Naive', top_k=2, store='vector', model_name='Salesforce/codet5-base', **kwargs):
        """
        Initialize the Retriever class.

        Args:
            chunking (str): The chunking method to use.
            top_k (int): Number of top results to retrieve.
            store (str): Type of storage (e.g., 'vector').
            model_name (str): Pretrained model name (e.g., 'Salesforce/codet5-base').
        """
        self.chunking = chunking
        self.top_k = top_k
        self.store = store
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.knowledge_base = {}

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model outputs.

        Args:
            model_output: Output from the transformer model.
            attention_mask: Attention mask to exclude padding tokens.

        Returns:
            Tensor: Pooled sentence embeddings.
        """
        token_embeddings = model_output.last_hidden_state  # Extract token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def chunk_github_repo(self, GITHUB_REPOS, chunk_size=200):
        """
        Clone and chunk the content of a GitHub repository.

        Args:
            GITHUB_REPO (str): The URL of the GitHub repository.
            chunk_size (int): Number of words per chunk.
        """
        GITHUB_TOKEN = "github_token_here"
        # Example for downloading raw files (you can expand this to clone a full repo)
        code_files_urls = crawl_multiple_repos(GITHUB_REPOS)

        chunks = []

        for i in range(0, len(code_files_urls)):
            if code_files_urls[i].endswith(".py"):
                content = extract_python_code_from_py(code_files_urls[i])
                chunks.append(content)

        # Store chunks and their embeddings
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
            #print(inputs)
            outputs = self.model.encoder(**inputs)
            embedding = self.mean_pooling(outputs, inputs['attention_mask'])
            self.knowledge_base[chunk] = embedding.detach()

    def retrieve_top_k_chunks(self, query, k=1):
        """
        Retrieve the top-k relevant chunks for a given query.

        Args:
            query (str): The query string.
            k (int): Number of top chunks to retrieve.

        Returns:
            List[str]: The top-k relevant chunks.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        query_embedding = self.mean_pooling(self.model.encoder(**inputs), inputs['attention_mask'])

        scores = {
            chunk: util.pytorch_cos_sim(query_embedding, emb).item()
            for chunk, emb in self.knowledge_base.items()
        }

        # Sort chunks by similarity score in descending order
        top_chunks = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        return top_chunks

    

class AutoRAG:
    """
    A class for Retrieval-Augmented Generation using a specified model and dataset.
    """

    def __init__(self, model_name, dataset, rag = True,embedding_model= 'Salesforce/codet5-base', **kwargs): 
        self.model_name = model_name
        self.dataset = load_dataset(dataset)
        self.pipeline = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        self.retriever = Retriever(
            model_name = embedding_model
        )
        self.rag = rag

    def __repr__(self): 
        return f"{_ for _ in self.__dict__.keys()}"


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

        augmented_prompt = prompt

        # Retrieve top-k relevant chunks
        if self.rag:
            retrieved_chunks = self.retriever.retrieve_top_k_chunks(prompt, k=k)
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

        if self.rag:
            repo_url = [ 'haoel/leetcode'] #, 'haoel/leetcode'
            if repo_url:
                print(f"Processing GitHub repository from {repo_url}...")
                self.retriever.chunk_github_repo(repo_url)
         
                #print(f"Repository content stored in knowledge base. Total chunks: {len(self.knowledge_base)}")
        k = 1
        all_results = []
        '''
        for problem in self.dataset["test"]:
            results = self._evaluate_problem(problem, k=k)
            all_results.append(results)

        pass_at_k = self._calculate_pass_at_k(all_results, k=k)
        print(f"Pass@{k}: {pass_at_k:.4f}")
        '''

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





