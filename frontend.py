import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_path = "./Llama-3.2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Ensure model is on the correct device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate a legal-related answer
def generate_legal_answer(question, max_length=300):
    # Predefined legal-specific context (hidden from the user)
    context = (
        "You are a knowledgeable, unbiased, and ethical assistant specializing in legal matters. "
        "Always provide accurate and lawful information in a respectful and neutral tone. "
        "Your answers should adhere to the principles of legality and fairness, avoiding any harmful, "
        "discriminatory, or unethical content. If the question is vague, irrelevant, or beyond the scope "
        "of your knowledge, politely clarify or explain why an answer cannot be provided. "
        "Ensure all responses align with applicable laws and regulations.\n\n"
    )
    
    # Combine context with the user's question
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
    
    # Extract and return only the answer part
    answer_start = response.find("Answer:") + len("Answer:")
    answer = response[answer_start:].strip()
    
    return answer

# Streamlit UI
st.title("Legal Assistant - AI-Driven Legal Query Answering")

# Instruction for the user
st.markdown("### Please enter your legal question below:")

# Input from the user
user_question = st.text_area("Your Question:", placeholder="Enter your legal query here...")

if st.button("Generate Answer"):
    if user_question.strip():
        with st.spinner("Generating answer..."):
            answer = generate_legal_answer(user_question)
        st.success("Answer Generated!")
        st.markdown(f"### Generated Answer:\n{answer}")
    else:
        st.error("Please enter a valid question!")
