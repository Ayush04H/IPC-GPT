import pandas as pd
import json

# Load the dataset
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv(r'D:\Project\Dataset\ipc_sections.csv')

# Specify the output file
output_file = r'Dataset\formatted_dataset.jsonl'

# Open the JSONL file for writing
with open(output_file, 'w') as f:
    for _, row in data.iterrows():
        # Formulate the question and answer
        question = f"What is the punishment for {row['Offense']}?"
        answer = f"{row['Punishment']} (Section: {row['Section']})"
        
        # Create the dictionary structure
        qa_entry = {
            "question": question,
            "answer": answer
        }
        
        # Write each entry as a JSON line
        f.write(json.dumps(qa_entry) + '\n')

print(f"Data has been formatted and saved to {output_file}")
