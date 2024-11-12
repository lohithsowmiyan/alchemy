from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rouge_score import rouge_scorer
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the Natural Questions dataset (test split)
dataset = load_dataset("sentence-transformers/natural-questions")

# Define the model and tokenizer from Hugging Face
model_name = "meta-llama/Llama-3.2-1B"  # Replace with your specific model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up a text generation pipeline with GPU support
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Use the last 1000 examples for evaluation
dataset = dataset['train']

total_rouge1 = 0.0
total_rouge2 = 0.0
total_rougeL = 0.0
count = 0

# Perform inference and calculate ROUGE scores for each question
for idx, example in enumerate(dataset):
    if count > 10000: break
    question = example['query']  # Adjust the key if needed
    ground_truth_answer = example['answer']  # Adjust the key if needed

    print(f"Question {idx+1}: {question}")

    # Get the model's response using Hugging Face
    response = generator(question, max_length=50, num_return_sequences=1)
    model_answer = response[0]['generated_text']

    # Compute ROUGE scores
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

