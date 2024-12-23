from transformers import pipeline
import docx
import pandas as pd

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def generate_qa_pairs(text, qa_model):
    questions_answers = []
    chunks = text.split("\n")
    for chunk in chunks:
        if chunk.strip():  # Skip empty lines
            try:
                input_text = f"generate question: {chunk}"  # Formatting input for question generation
                result = qa_model(input_text, max_length=512, num_return_sequences=1)
                question = result[0]["generated_text"]  # Extract the generated question
                questions_answers.append({"question": question, "answer": chunk})
            except Exception as e:
                print(f"Error generating QA for chunk: {e}")
    return questions_answers

def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# Main script
if __name__ == "__main__":
    docx_file = r"D:\Project\Dataset\docx\2022080523.docx"
    output_csv = "output.csv"

    # Step 1: Extract text from .docx
    document_text = extract_text_from_docx(docx_file)

    # Step 2: Load a pre-trained QA model
    qa_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

    # Step 3: Generate questions and answers
    qa_pairs = generate_qa_pairs(document_text, qa_pipeline)

    # Step 4: Save to CSV
    save_to_csv(qa_pairs, output_csv)
    print(f"QA pairs saved to {output_csv}")
