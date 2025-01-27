import json
import csv

# Define input JSONL file path and output CSV file path
input_file_path = 'Dataset/formatted_dataset.jsonl'
output_file_path = 'Dataset/formatted_dataset.csv'

# Open the JSONL file and the CSV file for writing
with open(input_file_path, 'r') as jsonl_file, open(output_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header for the CSV
    csv_writer.writerow(['question', 'answer'])
    
    # Read each line in the JSONL file
    for line in jsonl_file:
        # Parse JSON object from line
        data = json.loads(line)
        
        # Write question and answer to CSV
        csv_writer.writerow([data['question'], data['answer']])

print(f"CSV file has been created successfully at: {output_file_path}")
