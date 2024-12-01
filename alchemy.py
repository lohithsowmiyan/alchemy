from datasets import load_dataset
from transformers import pipeline
import torch

class AutoRAG:
    """


    """

    def __init_(self, model_name, dataset, **kwargs): 

        self.model_name = model_name
        self.dataset = dataset

        self.pipeline = pipeline("text-generation", model_name = model_name, device = 0 if torch.cuda.is_available() else -1)


    def __repr__(self):
        return f"{_ for _ in self.keys()}"

    # Define the pass@k metric calculation
    def calculate_pass_at_k(results: List[List[bool]], k: int) -> float:
        """
        Calculate pass@k for a list of results.
        
        Args:
            results (List[List[bool]]): A list where each entry corresponds to the results for one prompt.
                                        Each entry is a list of booleans indicating whether each completion passed.
            k (int): The number of samples considered.
        
        Returns:
            float: The pass@k score.
        """
        num_problems = len(results)
        pass_count = 0
        
        for problem_results in results:
            pass_count += any(problem_results[:k])
        
        return pass_count / num_problems

    
    # Function to evaluate a single problem
    def evaluate_problem(problem: dict, k: int = 3) -> List[bool]:
        """
        Evaluate a single problem in the HumanEval dataset.
        
        Args:
            problem (dict): A single problem from the HumanEval dataset.
            k (int): Number of samples to generate.
        
        Returns:
            List[bool]: A list of booleans indicating if each generated completion passed the test.
        """
        prompt = problem["prompt"]
        test_code = problem["test"]
        language = problem["language"]

        # Generate k completions
        completions = generation_pipeline(prompt, num_return_sequences=k, max_length=512, temperature=0.7)

        results = []
        for completion in completions:
            code = completion["generated_text"]
            try:
                # Combine prompt and generated code
                exec_code = prompt + code
                
                # Execute the combined code
                local_env = {}
                exec(exec_code, {"__builtins__": None}, local_env)
                
                # Run the provided tests
                exec(test_code, {"__builtins__": None}, local_env)
                
                results.append(True)
            except Exception as e:
                results.append(False)
        
        return results




