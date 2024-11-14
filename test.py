from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up the paths
model_path = "./Llama-3b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure model is on the correct device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to generate answers
def generate_answer(prompt, max_length=50):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the generated response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Example usage
prompt = "What is IPC in Law?Give its Full Form"
answer = generate_answer(prompt)
print("Generated Answer:", answer)
