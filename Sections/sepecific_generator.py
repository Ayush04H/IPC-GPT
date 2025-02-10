import os
import google.generativeai as genai
from docx import Document

# Configure API key
genai.configure(api_key="")
# Create the model
generation_config = {
    "temperature": 1.05,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Automate generation for a list of IPC sections with descriptions
def generate_and_save_ipc_descriptions(sections, save_dir=None):
    """
    Generates descriptions for a list of IPC sections, including their names and statements.

    Args:
        sections (list): A list of dictionaries, where each dictionary contains
                       the section number and the section title.  Example:
                       [{'number': 1, 'title': 'Title and extent of operation of the Code.'},
                        {'number': 2, 'title': 'Punishment of offences committed within India.'}]
        save_dir (str, optional): The directory to save the generated documents.
                                   Defaults to the current working directory.
    """

    # Set save directory or use current directory
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it doesn't exist
    else:
        save_dir = os.getcwd()  # Use current directory

    chat_session = model.start_chat(history=[])
    for section in sections:
        section_number = section['number']
        section_title = section['title']

        # Generate prompt
        prompt = f"Describe IPC section {section_number}: '{section_title}' according to Indian Penal Code in 1000 words strictly"

        # Generate response
        print(f"Generating content for IPC Section {section_number}: {section_title}...")
        response = chat_session.send_message(prompt)
        content = response.text

        # Sanitize the section title for use in the filename
        safe_section_title = "".join(c if c.isalnum() or c in (' ', '_', '-') else "_" for c in section_title)
        safe_section_title = safe_section_title[:50]  # Truncate for filename length

        # Save to Word document
        file_name = os.path.join(save_dir, f"IPC Section {section_number} - {safe_section_title}.docx")
        document = Document()
        document.add_heading(f"IPC Section {section_number}: {section_title}", level=1)
        document.add_paragraph(content)
        document.save(file_name)
        print(f"Saved content for IPC Section {section_number}: {section_title} to {file_name}")


# Define the list of sections to process
sections_to_generate = [
    {'number': 205, 'title': 'False personation for purpose of act or proceeding in suit or prosecution.'},
    {'number': 206, 'title': 'Fraudulent removal or concealment of property to prevent its seizure as forfeited or in execution.'},
    {'number': 207, 'title': 'Fraudulent claim to property to prevent its seizure as forfeited or in execution.'},
    {'number': 208, 'title': 'Fraudulently suffering decree for sum not due.'},
    {'number': 209, 'title': 'Dishonestly making false claim in Court.'},
    {'number': 210, 'title': 'Fraudulently obtaining decree for sum not due.'},
    {'number': 211, 'title': 'False charge of offence made with intent to injure.'},
    {'number': 212, 'title': 'Harbouring offender.— if a capital offence; if punishable with imprisonment for life, or with imprisonment.'},
    {'number': 213, 'title': 'Taking gift, etc., to screen an offender from punishment.— if a capital offence; if punishable with imprisonment for life, or with imprisonment.'},
    {'number': 214, 'title': 'Offering gift or restoration of property in consideration of screening offenderif a capital offence; if punishable with imprisonment for life, or with imprisonment.'}
]
# You can now print the list to verify
# for section in sections_to_generate:
#     print(f"Section {section['number']}: {section['title']}")
# Call the function with the list of sections and a save directory
generate_and_save_ipc_descriptions(sections_to_generate, save_dir="Sections_2")