from datasets import load_dataset
from transformers import pipeline
import torch

class AutoRAG:
    """


    """
    def __init__(self, model_name, dataset, **kwargs): 

        self.model_name = model_name
        self.dataset = load_dataset(dataset)
        self.pipeline = pipeline("text-generation", model_name = model_name, device = 0 if torch.cuda.is_available() else -1)


    def __repr__(self): 
        return f"{_ for _ in self.keys()}"

    # Define the pass@k metric calculation
    @staticmethod
    def _calculate_pass_at_k(self, results: list[list[bool]], k: int) -> float:
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


    def _evaluate_problem(self, problem: dict, k: int = 3) -> list[bool]:
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
        

        # Generate k completions
        completions = self.pipeline(prompt, num_return_sequences=k, max_length=512, temperature=0.7)

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

    def benchmark(self):

        k = 1
        all_results = []
        for problem in self.dataset["test"]:
            results = self._evaluate_problem(problem, k=k)
            all_results.append(results)

        pass_at_k = self._calculate_pass_at_k(all_results, k=k)
        print(f"Pass@{k}: {pass_at_k:.4f}")

    
if __name__ == "__main__":

    auto = AutoRAG(
        model_name = "EleutherAI/gpt-neo-1.3B",
        dataset = "openai_humaneval",
    )

    auto.benchmark()





