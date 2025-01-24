from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load the LoRA model
model_path = "lora_model"

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    device_map="auto"
)

# CRITICAL STEP: Enable inference mode
model = FastLanguageModel.for_inference(model)

# Prepare text streamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Prepare query
messages = [
    {"role": "user", "content": "Under Indian law, what are the penalties for violating Civil codes?"}
]

# Tokenize input
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate response
output = model.generate(
    input_ids=inputs,
    streamer=text_streamer,
    max_new_tokens=1048,
    temperature=0.7
)