import torch
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
from tqdm import tqdm

# Load LoRA fine-tuned model and tokenizer
model_path = "lora_model"  # Your model folder path
load_in_4bit = True  # Whether to load in 4-bit precision

# Load the model
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if not load_in_4bit else torch.float32,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Enable inference mode (if necessary)
model.eval()

# Load the dataset
dataset_path = r"Dataset/finetunned/formatted_dataset.csv"  # Your dataset path
df = pd.read_csv(dataset_path)

# Check if 'question' column exists (correct column name)
if 'question' not in df.columns:
    raise ValueError("The CSV file must contain a 'question' column.")


def generate_answer(question):
    """Generates an answer for a given question using the LoRA model."""
    messages = [
        {"role": "user", "content": question}
    ]

    # Tokenize input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate response
    with torch.no_grad():  # Disable gradient calculation during inference
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            use_cache=True,
            temperature=1.0,
            min_p=0.1
        )

    # Decode the output
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the prompt from the generated answer
    prompt_length = len(tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )[0])

    # Extract the response (remove the prompt)
    response = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return response.strip()  # Remove leading/trailing whitespace


# Add a new column 'fine_tunned_model_answers' to the DataFrame (correct column name)
df['fine_tunned_model_answers'] = ""

# Iterate through each question and generate an answer
for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
    question = row['question']
    answer = generate_answer(question)
    df.loc[index, 'fine_tunned_model_answers'] = answer

# Save the updated DataFrame to a new CSV file
output_path = r"Dataset/finetunned/finetunned_dataset_with_answers.csv"
df.to_csv(output_path, index=False)

print(f"Answers generated and saved to {output_path}")