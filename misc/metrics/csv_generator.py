import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_path = "./Llama-3.2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Use CPU if CUDA fails due to memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_legal_answer(question, max_length=150):
    context = "You are a legal expert assistant providing ethical and lawful information."
    prompt = f"{context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_length=max_length, 
            temperature=0.7,
            top_p=0.95,
        )
        
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_start = response.find("Answer:") + len("Answer:")
    return response[answer_start:].strip()

df = pd.read_csv(r'Dataset\finetunned\analysis_dataset.csv')
if 'Answers' not in df.columns:
    df['Answers'] = ""

for i, row in df.iterrows():
    question = row['Question']
    if not pd.isna(question):
        print(f"Generating answer for question {i + 1}: {question}")
        df.at[i, 'Answers'] = generate_legal_answer(question)

df.to_csv(r'Dataset\finetunned\analysis_dataset_copy.csv', index=False)
print("Updated CSV saved.")
