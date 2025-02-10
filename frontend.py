import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

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

# Prepare the prompt
messages = [
    {"role": "user", "content": "What is the motor vehicle act?"}
]

# Tokenize input
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Use a text streamer for efficient streaming output
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Generate response
output = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=256,
    use_cache=True,
    temperature=1.0,
    min_p=0.1
)

print("Generation Complete!")