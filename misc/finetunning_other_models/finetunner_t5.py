import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

# Set environment variable to avoid duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the dataset
dataset = load_dataset("json", data_files="Dataset/formatted_dataset.jsonl")

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenize the data with correct input formatting
def preprocess_function(examples):
    questions = examples["question"]
    answers = examples["answer"]

    # Format the inputs as "<question> <answer>"
    inputs = [f"question: {q} answer: {a}" for q, a in zip(questions, answers)]
    
    # Tokenize the inputs and outputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(answers, max_length=512, truncation=True, padding="max_length")

    # Set the labels for training
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Map preprocessing function to the dataset
tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# Split the dataset into training and validation sets
train_test_split = tokenized_data["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define training arguments
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
    eval_steps=50,
    predict_with_generate=True  # This is important for sequence-to-sequence tasks
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
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuning completed and model saved.")

# Test the model on a sample question-answering task
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Define a sample question for testing the model
sample_question = "What is the punishment for wearing the dress or carrying any token used by a soldier, sailor or airman with intent that it may be believed that he is such a soldier, sailor or airman?"

# Generate and print the answer for the sample question
result = qa_pipeline(f"question: {sample_question}")

# Print the answer
print("Generated Answer:", result[0]['generated_text'])
