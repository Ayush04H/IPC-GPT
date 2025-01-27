import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set up the model path and device
model_path = "./Llama-3b"  # Path to your local LLaMA model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Define the function to generate answers from the model
def generate_answer(description, offense, max_length=450):
    # Refine the prompt using both Description and Offense
    refined_prompt = f"Description: {description}\nOffense: {offense}\nProvide only the punishment details in a short, complete sentence. Answer:"
    
    # Tokenize the prompt
    inputs = tokenizer(refined_prompt, return_tensors="pt").to(device)
    
    # Generate response with adjusted parameters
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=5,            # Further lowered for more focused output
            top_p=0.8,          # Reduced randomness for predictable responses
            temperature=0.5,    # Lowered temperature for less creative output
            repetition_penalty=1.2,  # Penalize repeating parts of the question
            early_stopping=True  # Stop when a coherent answer is generated
        )
    
    # Decode the generated response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Post-process to remove any trailing text from the question
    answer = answer.replace(refined_prompt, "").strip()  # Strip prompt from answer
    return answer

# Path to the input CSV file
output_file_path = 'Temperatures/05/ipc_sections.csv'

# Load the CSV file
df = pd.read_csv(output_file_path)

# Ensure required columns are present
if 'Description' not in df.columns or 'Offense' not in df.columns or 'Actual Punishment' not in df.columns:
    raise ValueError("CSV file must contain 'Description', 'Offense', and 'Actual Punishment' columns.")

# Add a new column for LLaMA responses (Predicted Punishment)
df['Actual Punishment'] = df.apply(lambda row: generate_answer(row['Description'], row['Offense']), axis=1)

# Save the updated DataFrame back to the CSV
df.to_csv(output_file_path, index=False)
print(f"LLaMA answers added to '{output_file_path}'")

# Now, let's calculate confusion matrix metrics
# For simplicity, let's assume the actual and predicted answers are categorical (e.g., specific punishments)
# If you have free-text answers, you'll need a mechanism to match these predictions to predefined categories

# Define a function to preprocess the answers for classification (categorical)
def categorize_answer(answer):
    # You can define your categorization logic here, for example:
    if 'imprisonment' in answer:
        return 'Imprisonment'
    elif 'fine' in answer:
        return 'Fine'
    else:
        return 'Other'

# Apply categorization function
df['Predicted Category'] = df['Predicted Punishment'].apply(categorize_answer)
df['Actual Category'] = df['Actual Punishment'].apply(categorize_answer)

# Generate confusion matrix
y_true = df['Actual Category']
y_pred = df['Predicted Category']

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=['Imprisonment', 'Fine', 'Other'])

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Imprisonment', 'Fine', 'Other'], yticklabels=['Imprisonment', 'Fine', 'Other'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate and print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
