import streamlit as st
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# Load LoRA fine-tuned model and tokenizer
model_path = "lora_model"  # Your model folder path
load_in_4bit = True  # Whether to load in 4-bit precision

# Load the model
@st.cache_resource
def load_model():
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if not load_in_4bit else torch.float32,
        load_in_4bit=load_in_4bit,
        device_map="auto"
    )
    model.eval()
    return model

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(model_path)

model = load_model()
tokenizer = load_tokenizer()

def generate_response(question):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=1048,
        use_cache=True,
        temperature=0.7,
        min_p=0.1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("Indian Penal Code AI Assistant")

question = st.text_area("Ask a legal question:")
if st.button("Generate Response"):
    if question.strip():
        with st.spinner("Generating response..."):
            answer = generate_response(question)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
