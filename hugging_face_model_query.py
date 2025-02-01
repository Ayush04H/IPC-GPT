import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Load model from Hugging Face Hub
base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3.2-3b-instruct-bnb-4bit")
model = PeftModel.from_pretrained(base_model, "ayush0504/Fine-Tunned-GPT")
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ayush0504/Fine-Tunned-GPT")

def generate_response(question):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=1048,
        use_cache=True,
        temperature=0.7,
        min_p=0.1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    question = input("Ask a legal question: ")
    if question.strip():
        answer = generate_response(question)
        print("\nAnswer:", answer)
    else:
        print("Please enter a valid question.")
