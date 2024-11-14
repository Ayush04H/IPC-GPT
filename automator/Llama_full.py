import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up the model path and device
model_path = "./Llama-3b"  # Path to your local LLaMA model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Define the function to generate answers from the model
def generate_answer(description, offense, max_length=450):
    # Refine the prompt using both Description and Offense
    refined_prompt = f"Description: {description}\nOffense: {offense}\nProvide only the punishment details in a short, complete sentence. Answer:"
    
    # Tokenize the prompt
    inputs = tokenizer(refined_prompt, return_tensors="pt").to(device)
    
    # Generate response with adjusted parameters
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=5,            # Further lowered for more focused output
            top_p=0.8,          # Reduced randomness for predictable responses
            temperature=0.2,    # Lowered temperature for less creative output
            repetition_penalty=1.2,  # Penalize repeating parts of the question
            early_stopping=True  # Stop when a coherent answer is generated
        )
    
    # Decode the generated response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Post-process to remove any trailing text from the question
    answer = answer.replace(refined_prompt, "").strip()  # Strip prompt from answer
    return answer

# Path to the input CSV file
output_file_path = 'Dataset/ipc_sections_copy.csv'

# Load the CSV file
df = pd.read_csv(output_file_path)

# Ensure required columns are present
if 'Description' not in df.columns or 'Offense' not in df.columns:
    raise ValueError("CSV file must contain 'Description' and 'Offense' columns.")

# Add a new column for LLaMA responses
df['Predicted Punishment'] = df.apply(lambda row: generate_answer(row['Description'], row['Offense']), axis=1)

# Save the updated DataFrame back to the CSV
df.to_csv(output_file_path, index=False)
print(f"LLaMA answers added to '{output_file_path}'")
