from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

# Check if CUDA is available and use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the LoRA model
model_path = "lora_model"

# Load model with device map explicitly set
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    device_map="auto"  # Automatically maps to available GPUs
)

# Ensure the model is moved to inference mode
model = FastLanguageModel.for_inference(model)

# Move model to GPU (if not already on the GPU)
model = model.to(device)

# Prepare text streamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Prepare query
messages = [
    {"role": "user", "content": "Under Indian law, what are the penalties for violating Civil codes?"}
]

# Tokenize input and move to the correct device
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)  # Move input to the GPU

# Generate response
output = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=1048,
    temperature=0.7
)

# Display the output
print("Generated response:", output)
