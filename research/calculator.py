import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
import pandas as pd
import re
from typing import List, Dict
import difflib
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Model Loading & Setup
def load_model_and_tokenizer(model_path: str, load_in_4bit: bool = True):
    """Loads the fine-tuned model and tokenizer."""
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if not load_in_4bit else torch.float32,
            load_in_4bit=load_in_4bit,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        logging.info(f"Model and tokenizer loaded successfully from {model_path}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model/tokenizer: {e}")
        raise

# 2. Inference Function (Unchanged)
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generates a response from the model given a prompt."""
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        output = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=1.0,
            min_p=0.1
        )

        # Decode the output and clean up special tokens
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the input prompt from the generated text to get a clean response
        cleaned_response = generated_text[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
        logging.info("Response generated successfully.")
        return cleaned_response

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return None

# 3. Fact-Checking with CSV (Revised)
def fact_check(response: str, actual_answer: str, similarity_threshold: float = 0.8):
    """
    Fact-checks the model's response against the actual answer.

    Args:
        response (str): The model's generated response.
        actual_answer (str): The actual answer from the CSV.
        similarity_threshold (float): Minimum similarity score to consider a match.

    Returns:
        tuple: (bool, str) - True if the response is similar to the actual answer, False otherwise, and a reason.
    """
    try:
        # Calculate similarity between the generated response and the actual answer
        similarity_score = calculate_similarity(response, actual_answer)

        if similarity_score >= similarity_threshold:
            return True, f"Response is similar to actual answer (similarity: {similarity_score:.2f})"
        else:
            return False, f"Response is dissimilar to actual answer (similarity: {similarity_score:.2f})"

    except Exception as e:
        logging.error(f"Error during fact-checking: {e}")
        return False, f"Error during fact-checking: {e}"


# 4. Hallucination Detection (Slightly Revised)
def detect_hallucinations(response: str, fact_check_result: bool, fact_check_reason: str):
    """Detects potential hallucinations based on fact-checking and response content."""
    hallucinations = []

    if not fact_check_result:
        hallucinations.append(f"Hallucination: Fact-checking failed. Reason: {fact_check_reason}")

    # Look for patterns suggesting fabricated legal terms (Improve this REGEX!)
    if re.search(r"legal term .* does not exist", response, re.IGNORECASE):
        hallucinations.append("Hallucination: Mentions a non-existent legal term.")

    # Check for contradictions (This requires more advanced NLP)
    # Placeholder: Implement contradiction detection logic.

    return hallucinations

# 5. Calculate Similarity
def calculate_similarity(text1: str, text2: str):
    """Calculates the similarity between two texts using difflib."""
    try:
        seq_matcher = difflib.SequenceMatcher(None, text1, text2)
        similarity_score = seq_matcher.ratio()
        return similarity_score
    except Exception as e:
        logging.error(f"Error calculating similarity: {e}")
        return 0.0

# 6. Evaluation Framework (Revised to handle your CSV format)
def evaluate_model(model, tokenizer, csv_path: str, max_new_tokens: int = 256):
    """Evaluates the model on a dataset of questions and answers from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Ensure the CSV has the correct columns
        if 'Questions' not in df.columns or 'Actual Answer' not in df.columns or 'Generated Answer' not in df.columns:
            raise ValueError("CSV file must contain 'Questions', 'Actual Answer', and 'Generated Answer' columns.")
    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_path}")
        return
    except ValueError as e:
        logging.error(e)
        return
    except Exception as e:
        logging.error(f"Error loading or validating CSV file: {e}")
        return

    results = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Queries"):
        question = row['Questions']
        actual_answer = row['Actual Answer']
        generated_answer = row['Generated Answer']  # Get existing generated answer

        if pd.isna(generated_answer) or not generated_answer:  # Check if it's empty
            generated_answer = generate_response(model, tokenizer, question, max_new_tokens)
            df.loc[index, 'Generated Answer'] = generated_answer  # Store generated answer back in DataFrame

        if generated_answer:
            fact_check_result, fact_check_reason = fact_check(generated_answer, actual_answer)
            hallucinations = detect_hallucinations(generated_answer, fact_check_result, fact_check_reason)

            results.append({
                'question': question,
                'actual_answer': actual_answer,
                'generated_answer': generated_answer,
                'fact_check_result': fact_check_result,
                'fact_check_reason': fact_check_reason,
                'hallucinations': hallucinations
            })
        else:
            results.append({
                'question': question,
                'actual_answer': actual_answer,
                'generated_answer': None,
                'fact_check_result': False,
                'fact_check_reason': "Failed to generate response.",
                'hallucinations': ["Failed to generate response."]
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    df.to_csv("updated_qa_dataset.csv", index=False)  # Save updated CSV with generated answers
    logging.info("Evaluation complete. Results saved to evaluation_results.csv and updated_qa_dataset.csv")

    # Basic analysis of results (can be expanded)
    hallucination_count = sum(len(result['hallucinations']) > 0 for result in results)
    print(f"Total queries: {len(df)}")
    print(f"Number of queries with detected hallucinations: {hallucination_count}")

# 7. Mitigation Strategies (Unchanged Outline)
def mitigate_hallucinations(query: str, response: str, hallucinations: List[str]):
    """Applies mitigation strategies to reduce hallucinations."""
    # Implement specific mitigation techniques based on the detected hallucinations.
    pass

# 8. Main Execution
if __name__ == "__main__":
    model_path = "lora_model"  # Your model folder path
    csv_path = r"D:\Project\dataset_preparer\Evaluation_Metric_dataset.csv"  # Path to your CSV with questions and answers

    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
        evaluate_model(model, tokenizer, csv_path)
        print("Research script completed.")
    except Exception as e:
        print(f"An error occurred: {e}")