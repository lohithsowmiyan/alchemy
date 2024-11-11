from datasets import load_dataset
import ollama
from rouge_score import rouge_scorer

# Load the Natural Questions dataset (test split)
dataset =  load_dataset("sentence-transformers/natural-questions")



# Set up the Ollama API to use the model
ollama_model = "llama3.2"  # Replace with the exact model name if different

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

print(dataset)

# Perform inference and calculate ROUGE-L for each question
for idx, example in enumerate(dataset['train']):
    question = example['query']  # Adjust the key if needed
    ground_truth_answer = example['answer']  # Adjust the key if needed

    print(f"Question {idx+1}: {question}")
    
    # Get the model's response using Ollama
    response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": question}])
    model_answer = response['message']['content']
    #print(model_answer)  # Extract the model's generated answer

    # Compute ROUGE-L score
    scores = scorer.score(ground_truth_answer, model_answer)
    rougeL_score = scores['rougeL'].fmeasure  # ROUGE-L F-measure score

    print(f"Ground Truth Answer: {ground_truth_answer}")
    print(f"Model Answer: {model_answer}")
    print(f"ROUGE-L Score: {rougeL_score:.4f}\n")
