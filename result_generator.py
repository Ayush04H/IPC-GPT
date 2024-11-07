import os
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline

# Set environment variable to avoid duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the fine-tuned model and tokenizer
model_name = "./fine_tuned_bart_model"  # Path to your fine-tuned model directory
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize a text generation pipeline
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Define a function to get the generated answer
def generate_answer(question, context, max_answer_length=200):
    input_text = f"Question: {question} Context: {context}"
    result = qa_pipeline(input_text, max_length=max_answer_length, truncation=True)
    return result[0]['generated_text']

# Example usage
sample_context = """
According to section 127 of Indian penal code, Whoever receives any property knowing the same to have been taken in the commission of any of the offences mentioned in sections 125 and 126, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine and to forfeiture of the property so received.
"""
sample_question = "What is the punishment for Receiving property taken by war or depredation mentioned in sections 125 And 126?"

# Generate the answer
answer = generate_answer(sample_question, sample_context)
print("Generated Answer:", answer)
