from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Paths to the fine-tuned model
model_path = "./fine_tuned_llama_model"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set up the text-generation pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Sample context and question for testing
sample_context = """
According to section 140 of the Indian Penal Code, whoever, not being a soldier, sailor, or airman in the Military, Naval, or Air service of the Government of India, wears any garb or carries any token resembling any garb or token used by such a soldier, sailor, or airman with the intention that it may be believed that he is such a soldier, sailor, or airman, shall be punished with imprisonment of either description for a term which may extend to three months, or with a fine which may extend to five hundred rupees, or with both.
"""
sample_question = "What is the punishment for wearing the dress or carrying any token used by a soldier, sailor, or airman with intent that it may be believed that he is such a soldier, sailor, or airman?"

# Input format for the model
input_text = f"Context: {sample_context} Question: {sample_question} Answer:"

# Generate the answer
result = qa_pipeline(input_text, max_length=300, do_sample=True, truncation=True)

# Print the generated answer
print("Generated Answer:", result[0]['generated_text'])
