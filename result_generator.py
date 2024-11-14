from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Paths to the fine-tuned model
model_path = "./fine_tuned_llama_model"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Set up the text-generation pipeline
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Sample context and question for testing

sample_question = "What is the punishment for wearing the dress or carrying any token used by a soldier, sailor, or airman with intent that it may be believed that he is such a soldier, sailor, or airman?"

# Input format for the model
input_text = f"Question: {sample_question} Answer:"

# Generate the answer
result = qa_pipeline(input_text, max_length=300, do_sample=True, truncation=True)

# Print the generated answer
print("Generated Answer:", result[0]['generated_text'])
