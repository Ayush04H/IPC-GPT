import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import load_dataset

# Set environment variable to avoid duplicate library issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
model_path = "D:/Project/Llama-3b"  # Adjust to your local path
dataset_path = "Dataset/formatted_dataset.jsonl"

# Load the dataset
dataset = load_dataset("json", data_files=dataset_path)

# Load model and tokenizer for Llama using AutoTokenizer and AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Add padding token if missing (use EOS as padding token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Preprocess the dataset
# Preprocess the dataset with corrected shape alignment
def preprocess_function(examples):
    questions = examples["question"]
    answers = examples["answer"]

    # Concatenate question and answer for decoder-based training
    inputs = [f"Question: {q} Answer:" for q in questions]
    targets = [a for a in answers]

    # Tokenize the inputs and targets with consistent padding and truncation
    model_inputs = tokenizer(
        inputs,
        max_length=512,  # Adjust max_length as needed
        padding="max_length",  # Ensure padding for inputs
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize the answers (targets) for the model with consistent padding
    labels = tokenizer(
        targets,
        max_length=512,  # Ensure max_length matches with input sequence
        padding="max_length",  # Ensure padding for targets
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    # Replace padding token ids in the labels with -100 to ignore padding during loss calculation
    labels[labels == tokenizer.pad_token_id] = -100

    # Assign labels for supervised learning
    model_inputs["labels"] = labels
    return model_inputs


# Map preprocessing function to the dataset
tokenized_data = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# Split the dataset into training and validation sets
train_test_split = tokenized_data["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir='./llama_fine_tuned_results',
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduces memory usage per batch
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Accumulates gradients to simulate larger batch size
    fp16=True,  # Use mixed-precision training if supported
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./llama_logs',
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=100,  # Increases the interval for evaluation to save memory
    save_total_limit=2,  # Limits the number of saved model checkpoints
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
model.save_pretrained("./fine_tuned_llama_model")
tokenizer.save_pretrained("./fine_tuned_llama_model")

print("Fine-tuning completed and model saved.")

# Test the model on a sample question-answering task
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define a sample context and question for testing
sample_context = """
According to section 140 of Indian Penal Code, whoever, not being a soldier, sailor or airman in the Military, Naval or Air service of the Government of India, wears any garb or carries any token resembling any garb or token used by such a soldier, sailor or airman with the intention that it may be believed that he is such a soldier, sailor or airman, shall be punished with imprisonment of either description for a term which may extend to three months, or with fine which may extend to five hundred rupees, or with both.
"""
sample_question = "What is the punishment for wearing the dress or carrying any token used by a soldier, sailor or airman with intent that it may be believed that he is such a soldier, sailor or airman?"

# Generate and print the answer for the sample question
input_text = f"Question: {sample_question} Context: {sample_context} Answer:"
result = qa_pipeline(input_text, max_length=200, truncation=True)

print("Generated Answer:", result[0]['generated_text'])
