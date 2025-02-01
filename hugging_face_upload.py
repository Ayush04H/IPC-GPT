from huggingface_hub import HfApi

# Initialize the API object
api = HfApi()

# Define the path to your local folder and the Space repo ID
folder_path = "D:\Project\Test"  # Replace with the actual local path to your folder
repo_id = "ayush0504/LLM-CHATBOT"    # Replace with your Hugging Face Space repo

# Upload the folder
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="space",
)
