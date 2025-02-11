import os
import pandas as pd
from docx import Document

def process_word_file(input_file):
    # Define the context as a single line
    context = (
        "You are a knowledgeable, unbiased, and ethical assistant specializing in legal matters. Always provide accurate and lawful information in a respectful and neutral tone. Your answers should adhere to the principles of legality and fairness, avoiding any harmful, discriminatory, or unethical content. If the question is vague, irrelevant, or beyond the scope of your knowledge, politely clarify or explain why an answer cannot be provided. Ensure all responses align with applicable laws and regulations."
    )

    # Read the content of the Word document
    try:
        document = Document(input_file)
        content = " ".join([
            paragraph.text.replace("*", "").replace("** Text**", "").replace("#", "").replace("##", "").strip() 
            for paragraph in document.paragraphs 
            if paragraph.text.strip()
        ])
    except Exception as e:
        print(f"Error reading the Word file '{input_file}': {e}")
        return None

    # Extract IPC section number from the document
    section_number = "{Read IPC Section number}"  # Default placeholder if no section number found
    for line in content.split("."):
        if "IPC Section" in line:
            section_number = line.split("IPC Section")[-1].strip()
            break

    # Construct the question
    question = f"What does IPC section {section_number} describe? Explain in detail."

    # Combine context, question, and content
    combined_text = f"<s> [INST] <<SYS>> {context} {question} [/INST] {content} </s>"

    return combined_text

def process_all_word_files(folder_path, output_file):
    # Initialize the CSV file
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["text"]).to_csv(output_file, index=False)

    # Process each Word file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx"):
            input_file = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            combined_text = process_word_file(input_file)

            if combined_text:
                # Append to the CSV file
                df = pd.DataFrame({"text": [combined_text]})
                try:
                    df.to_csv(output_file, mode='a', header=False, index=False)
                    print(f"Added content from '{filename}' to the CSV file.")
                except Exception as e:
                    print(f"Error appending to the CSV file: {e}")

# Specify the folder containing Word files and the output CSV file
folder_path = "Sections_2"  # Replace with your actual folder path
output_file = r"dataset_preparer\Dataset_finetune_newdata.csv"

# Run the processing function
process_all_word_files(folder_path, output_file)