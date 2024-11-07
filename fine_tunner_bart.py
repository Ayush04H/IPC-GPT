import os
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

# Set environment variable to avoid duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the dataset
dataset = load_dataset("json", data_files="Dataset/formatted_dataset.jsonl")

# Load model and tokenizer for BART
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenize the data with context and set placeholder positions
def preprocess_function(examples):
    questions = examples["question"]
    answers = examples["answer"]

    # Tokenize questions and context for BART (sequence-to-sequence model)
    inputs = tokenizer(
        questions,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # Tokenize answers (targets) for BART
    targets = tokenizer(
        answers,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    inputs["labels"] = targets["input_ids"]
    
    return inputs

# Map preprocessing function to the dataset
tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# Split the dataset into training and validation sets
train_test_split = tokenized_data["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define training arguments with evaluation steps
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_bart_model")
tokenizer.save_pretrained("./fine_tuned_bart_model")

print("Fine-tuning completed and model saved.")

# Test the model on a sample question-answering task
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Define a sample context and question for testing the model
sample_context = """
According to section 140 of Indian Penal Code, whoever, not being a soldier, sailor or airman in the Military, Naval or Air service of the Government of India, wears any garb or carries any token resembling any garb or token used by such a soldier, sailor or airman with the intention that it may be believed that he is such a soldier, sailor or airman, shall be punished with imprisonment of either description for a term which may extend to three months, or with fine which may extend to five hundred rupees, or with both.
"""
sample_question = "What is the punishment for wearing the dress or carrying any token used by a soldier, sailor or airman with intent that it may be believed that he is such a soldier, sailor or airman?"

# Generate and print the answer for the sample question
input_text = f"Question: {sample_question} Context: {sample_context}"

result = qa_pipeline(
    input_text,
    max_length=200,  # Adjust max answer length for longer responses
    truncation=True
)

# Print the answer
print("Generated Answer:", result[0]['generated_text'])
