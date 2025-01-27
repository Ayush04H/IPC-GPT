from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up the paths
model_path = "./Llama-3.2-3b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure model is on the correct device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function to generate legal-related answers
def generate_legal_answer(question, max_length=300):
    # Legal-specific context
    context = (
        "You are a knowledgeable, unbiased, and ethical assistant specializing in legal matters. "
        "Always provide accurate and lawful information in a respectful and neutral tone. "
        "Your answers should adhere to the principles of legality and fairness, avoiding any harmful, "
        "discriminatory, or unethical content. If the question is vague, irrelevant, or beyond the scope "
        "of your knowledge, politely clarify or explain why an answer cannot be provided. "
        "Ensure all responses align with applicable laws and regulations.\n\n"
    )
    
    # Combine context with the question
    prompt = context + "Question: " + question + "\nAnswer:"
    
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
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the answer part
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer

# Example usage
question = (
    "Under Indian law, what are the penalties for violating traffic rules such as Breaking Signal On red light? "
    "Provide the relevant sections of the IPC or Motor Vehicles Act."
)
answer = generate_legal_answer(question)
print("Generated Answer:", answer)
