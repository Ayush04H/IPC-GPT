import csv
import random

def generate_legal_ai_dataset_csv(filename="legal_ai_dataset_25rows.csv", num_rows=25):
    """
    Generates a CSV dataset with 25 rows and 7 columns for legal AI hallucination analysis.
    Columns are: Questions, Actual Answer, Generated Answer, Is Hallucination, Is Correct Answer, Confidence Score, Model Type.

    Args:
        filename (str): The name of the CSV file to create (default: "legal_ai_dataset_25rows.csv").
        num_rows (int): The number of rows (data points) to generate in the dataset (default: 25).
    """

    header = ["Questions", "Actual Answer", "Generated Answer", "Is Hallucination", "Is Correct Answer", "Confidence Score", "Model Type"]
    data = []

    ipc_sections_examples = {
        "What is IPC section 302?": "Section 302 of the Indian Penal Code deals with punishment for murder.",
        "Define theft under IPC.": "Theft is defined under IPC Section 378.",
        "What is punishment for theft under IPC 379?": "Punishment for theft under IPC 379 is imprisonment up to 3 years or fine or both.",
        "Is self-defense allowed in IPC?": "Yes, self-defense is a valid defense under IPC Sections 96 to 106.",
        "What is 'grievous hurt' as per IPC?": "'Grievous hurt' is defined under IPC Section 320 and includes specific types of injuries.",
        "Explain 'criminal conspiracy' under IPC.": "'Criminal conspiracy' is defined under IPC Section 120A and deals with agreements to commit offenses.",
        "What is the punishment for 'cheating' under IPC?": "Punishment for 'cheating' under IPC Section 420 can be imprisonment and fine.",
        "Define 'abetment' in IPC.": "'Abetment' is explained in IPC Section 107 and refers to instigating or aiding an offense.",
        "What is 'false evidence' under IPC?": "'False evidence' is defined under IPC Section 191 and relates to giving untrue statements in judicial proceedings.",
        "Explain 'public nuisance' as per IPC.": "'Public nuisance' is defined under IPC Section 268 and covers acts causing common danger or annoyance."
    }

    model_types = ["Baseline", "Fine-Tuned"]

    for i in range(num_rows):
        question = random.choice(list(ipc_sections_examples.keys()))
        actual_answer = ipc_sections_examples[question]
        model_type = random.choice(model_types)

        # Simulate Generated Answer with some chance of hallucination
        if model_type == "Baseline":
            if random.random() < 0.4:  # 40% chance of hallucination for Baseline
                hallucinate = True
                correct_answer = False
                if random.random() < 0.5:
                    generated_answer = "IPC section 511A deals with this." # Fabricated section
                else:
                    generated_answer = "This is not a crime under IPC." # Incorrect interpretation
                confidence_score = round(random.uniform(0.6, 0.8), 2) # Lower confidence for baseline hallucinations
            else:
                hallucinate = False
                correct_answer = True
                generated_answer = actual_answer  # Baseline sometimes gets it right
                confidence_score = round(random.uniform(0.7, 0.95), 2) # Moderate to high confidence for baseline correct answers

        elif model_type == "Fine-Tuned":
            if random.random() < 0.1:  # 10% chance of hallucination for Fine-Tuned (lower)
                hallucinate = True
                correct_answer = False
                generated_answer = "According to IPC section 299A..." # Slightly plausible but wrong section
                confidence_score = round(random.uniform(0.5, 0.7), 2) # Lower confidence for fine-tuned hallucinations
            else:
                hallucinate = False
                correct_answer = True
                generated_answer = actual_answer # Fine-tuned mostly correct
                confidence_score = round(random.uniform(0.85, 0.99), 2) # Higher confidence for fine-tuned correct answers

        data_row = [question, actual_answer, generated_answer, hallucinate, correct_answer, confidence_score, model_type]
        data.append(data_row)

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
            csv_writer.writerows(data)
        print(f"CSV dataset file '{filename}' with {num_rows} rows generated successfully.")
    except Exception as e:
        print(f"Error generating CSV file: {e}")


if __name__ == "__main__":
    generate_legal_ai_dataset_csv() # Generates 25 rows by default
    # To generate a different number of rows, you can call the function with num_rows argument:
    # generate_legal_ai_dataset_csv(filename="legal_ai_dataset_50rows.csv", num_rows=50)