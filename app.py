import streamlit as st
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
import io
import sys
import threading
import time
import queue # Import the queue module

# --- Configuration ---
DEFAULT_MODEL_PATH = "lora_model" # Or your default path
DEFAULT_LOAD_IN_4BIT = True

# --- Page Configuration ---
st.set_page_config(page_title="Fine-tuned LLM Chat Interface", layout="wide")
st.title("Fine-tuned LLM Chat Interface")

# --- Model Loading (Cached) ---
@st.cache_resource(show_spinner="Loading model and tokenizer...")
def load_model_and_tokenizer(model_path, load_in_4bit):
    """Loads the PEFT model and tokenizer."""
    try:
        torch_dtype = torch.bfloat16 if load_in_4bit else torch.float16 # bfloat16 often better for 4-bit

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from path '{model_path}': {e}", icon="🚨")
        print(f"Error loading model: {e}")
        return None, None

# --- Custom Streamer Class (Modified for Queue) ---
class QueueStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt, q):
        super().__init__(tokenizer, skip_prompt=skip_prompt)
        self.queue = q
        self.stop_signal = None # Can be used if needed, but queue is primary

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Puts the text onto the queue."""
        self.queue.put(text)
        if stream_end:
             self.end()

    def end(self):
        """Signals the end of generation by putting None in the queue."""
        self.queue.put(self.stop_signal) # Put None (or a specific sentinel)


# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Model Configuration")
    st.info(f"Model loaded on startup: `{DEFAULT_MODEL_PATH}`, 4-bit: `{DEFAULT_LOAD_IN_4BIT}`.")

    st.header("Generation Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    min_p = st.slider("Min P", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    max_tokens = st.slider("Max New Tokens", min_value=50, max_value=2048, value=512, step=50)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Rerun to clear display immediately


# --- Load Model (runs only once on first run or if cache is cleared) ---
model, tokenizer = load_model_and_tokenizer(DEFAULT_MODEL_PATH, DEFAULT_LOAD_IN_4BIT)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Chat Interface ---
if model is None or tokenizer is None:
    st.error("Model loading failed. Please check the path and logs. Cannot proceed.")
    st.stop()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
user_input = st.chat_input("Ask the fine-tuned model...")

if user_input:
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare for model response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        text_queue = queue.Queue() # Create a queue for this specific response
        # Initialize the modified streamer
        text_streamer = QueueStreamer(tokenizer, skip_prompt=True, q=text_queue)

        # Prepare input for the model
        messages_for_model = st.session_state.messages

        try:
            if tokenizer.chat_template:
                 inputs = tokenizer.apply_chat_template(
                    messages_for_model,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                 ).to(model.device)
            else:
                 prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_for_model]) + "\nassistant:"
                 inputs = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

            # Generation arguments
            generation_kwargs = dict(
                input_ids=inputs,
                streamer=text_streamer, # Use the QueueStreamer
                max_new_tokens=max_tokens,
                use_cache=True,
                temperature=temperature if temperature > 0 else None,
                top_p=None,
                min_p=min_p,
                do_sample=True if temperature > 0 else False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            )

            # Define the target function for the thread
            def generation_thread_func():
                try:
                    # Run generation in the background thread
                    model.generate(**generation_kwargs)
                except Exception as e:
                    # If error occurs in thread, signal stop and maybe log
                    print(f"Error in generation thread: {e}")
                    text_streamer.end() # Ensure the queue loop terminates

            # Start the generation thread
            thread = threading.Thread(target=generation_thread_func)
            thread.start()

            # --- Main thread: Read from queue and update UI ---
            generated_text = ""
            while True:
                try:
                    # Get the next text chunk from the queue
                    # Use timeout to prevent blocking indefinitely if thread hangs
                    chunk = text_queue.get(block=True, timeout=1)
                    if chunk is text_streamer.stop_signal: # Check for end signal (None)
                        break
                    generated_text += chunk
                    response_placeholder.markdown(generated_text + "▌") # Update placeholder
                except queue.Empty:
                    # If queue is empty, check if the generation thread is still running
                    if not thread.is_alive():
                        # Thread finished, but maybe didn't put the stop signal (error?)
                        break # Exit loop
                    # Otherwise, continue waiting for next chunk
                    continue

            # Final update without the cursor
            response_placeholder.markdown(generated_text)

            # Add the complete assistant response to history *after* generation
            st.session_state.messages.append({"role": "assistant", "content": generated_text})

            # Wait briefly for the thread to finish if it hasn't already
            thread.join(timeout=2.0)


        except Exception as e:
            st.error(f"Error during generation setup or queue handling: {e}", icon="🔥")
            print(f"Error setting up generation or handling queue: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"*Error generating response: {e}*"})
            response_placeholder.error(f"Error generating response: {e}")