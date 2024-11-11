from datasets import load_dataset
import ollama
from rouge_score import rouge_scorer

# Load the Natural Questions dataset (test split)
dataset =  load_dataset("sentence-transformers/natural-questions")



# Set up the Ollama API to use the model
ollama_model = "llama3.2"  # Replace with the exact model name if different

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

dataset = dataset['train'][-1000:]

total_rouge1 = 0.0
total_rouge2 = 0.0
total_rougeL = 0.0
count = 0

# Perform inference and calculate ROUGE-L for each question
for idx, example in enumerate(dataset):
    question = example['query']  # Adjust the key if needed
    ground_truth_answer = example['answer']  # Adjust the key if needed

    print(f"Question {idx+1}: {question}")
    
    # Get the model's response using Ollama
    response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": question}])
    model_answer = response['message']['content']
    #print(model_answer)  # Extract the model's generated answer

    # Compute ROUGE-L score
    scores = scorer.score(ground_truth_answer, model_answer)
    total_rouge1 += scores['rouge1'].fmeasure
    total_rouge2 += scores['rouge2'].fmeasure
    total_rougeL += scores['rougeL'].fmeasure
    count += 1

avg_rouge1 = total_rouge1 / count
avg_rouge2 = total_rouge2 / count
avg_rougeL = total_rougeL / count

print(f"Average ROUGE-1 Score for last 1000 rows: {avg_rouge1:.4f}")
print(f"Average ROUGE-2 Score for last 1000 rows: {avg_rouge2:.4f}")
print(f"Average ROUGE-L Score for last 1000 rows: {avg_rougeL:.4f}")
