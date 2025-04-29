# Project Title: Fine-Tuning Language Models for Legal Question-Answering

## Overview
This project aims to fine-tune large language models, particularly Meta’s Llama-3B, for the task of question-answering in medical and legal contexts. The first stage focuses on analyzing dataset requirements, selecting the appropriate model, and evaluating generated answers using various similarity metrics. The second stage, currently in progress, will involve fine-tuning and deploying the model.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Evaluation Metrics](#evaluation-metrics)
- [Temperature Tuning Experiment](#temperature-tuning-experiment)
- [Results](#results)
- [Applications](#applications)
- [Stage 2: Fine-Tuning and Deployment](#stage-2-fine-tuning-and-deployment)
- [Usage](#usage)
- [References](#references)

## Project Structure
The project is organized as follows:
- `dataset/`: Contains the dataset files.
- `fine_tune.py`: Script for fine-tuning the model.
- `predict.py`: Script for generating predictions from the fine-tuned model.
- `README.md`: Project documentation.
- `requirements.txt`: List of required libraries.

## Dataset
The dataset consists of question-answer pairs related to medical and legal topics. The columns include:
- **question**: The question posed, such as legal or medical inquiries.
- **answer**: The corresponding answer, verified for accuracy and completeness.

### Dataset Curation and Preprocessing
- Text data was preprocessed to remove noise, including punctuation, special characters, and irrelevant text.
- Tokenization and normalization steps were applied to ensure consistency.
- For legal questions, context phrases such as "according to section" or "in law" were added to the training data to improve model accuracy.

## Model Selection
After comparing several models, **Meta's Llama-3B** was selected due to its performance on similar NLP tasks and its flexibility for fine-tuning on domain-specific tasks.

### Why Llama?
- Llama provides an optimal balance of performance and model size.
- Fine-tuning capabilities make it suitable for specific use cases in legal and medical NLP.
- Ability to handle long-form text, which is common in legal and medical contexts.

## Evaluation Metrics
To evaluate the similarity between the generated and actual answers, various metrics were considered:
1. **Cosine Similarity**: Measures the directional similarity between two text vectors, disregarding magnitude.
2. **Jaccard Similarity**: Compares sets of words, focusing on overlap rather than frequency.
3. **Euclidean Distance**: Measures the straight-line distance between vectors, considering magnitude differences.
4. **Manhattan Distance**: Sums the absolute differences between corresponding elements, capturing magnitude variations.
5. **Pearson Correlation**: Assesses the linear relationship between vectors, suitable for analyzing relationships.

Each metric provides unique insights, with cosine similarity generally favored for capturing semantic closeness, while Jaccard, Euclidean, Manhattan, and Pearson serve specific comparison needs in set overlap and magnitude-based analysis.

### Metric Selection Rationale
- **Cosine Similarity**: Chosen as the primary metric due to its focus on semantic closeness, making it ideal for evaluating model accuracy in capturing intent and meaning.
- **Jaccard Similarity**: Added to capture overlap in specific keywords or terminology, especially relevant in legal and medical language.
- **Euclidean and Manhattan Distances**: Useful for measuring the extent of divergence in overall structure, helping identify cases where generated answers differ significantly in style or length.

## Temperature Tuning Experiment
Temperature is a hyperparameter in language models that influences response randomness. Experiments were conducted at different temperatures (1.2, 0.8, 0.5, and 0.3) to find the optimal setting for generating concise, accurate answers. 

### Findings
- **Optimal Temperature (0.5)**: Best performance across metrics, balancing accuracy and coherence.
- **High Temperatures (1.2, 0.8)**: Increased diversity but lower accuracy, with responses often deviating from expected answers.
- **Low Temperature (0.3)**: Improved precision but led to repetitive and overly deterministic responses.

### Conclusion
A temperature of **0.5** was selected for optimal answer generation, as it provides the best balance between relevance and variability, particularly in complex legal and medical queries.

## Results
Based on initial experiments:
- **Cosine Similarity** yielded consistent results in identifying semantic alignment.
- **Jaccard Similarity** was sensitive to exact word matches, useful for token-level similarity.
- **Euclidean and Manhattan Distances** captured magnitude-based variations but lacked semantic depth.
- **Pearson Correlation** helped explore linear relationships between text vectors.

## Applications
This project has significant applications in the following areas:
1. **Medical Question-Answering Systems**: Assisting healthcare professionals by providing quick, accurate answers to medical queries.
2. **Legal Assistants**: Supporting legal professionals by offering case-based insights and rule interpretations.
3. **Domain-Specific Chatbots**: Enabling interactive systems capable of addressing specialized inquiries in medical and legal domains.
4. **Research Assistance**: Aiding in legal and medical research by providing fast access to relevant information, enhancing productivity for professionals.

## Stage 2: Fine-Tuning and Deployment
In the second stage, the model will be fine-tuned on the dataset, enabling it to generate accurate answers to domain-specific questions. The deployment phase will integrate the fine-tuned model into a question-answering pipeline.

### Deployment Plan
- **Integration with API**: The model will be hosted via an API endpoint for easy access in applications.
- **Real-Time Testing**: Deploying the model in a testing environment to monitor performance on real-world questions.
- **Scalability Considerations**: Optimizing for latency and scalability to handle increased traffic in production.

## Usage
1. **Fine-Tuning**:
   - Adjust the dataset path in `fine_tune.py`.
   - Run the script with the command:
     ```bash
     python fine_tune.py
     ```
2. **Prediction**:
   - After fine-tuning, use `predict.py` to generate answers for new questions.
   - Run the prediction script as follows:
     ```bash
     python predict.py --question "Your question here -- Question"
     ```

## References
- Hugging Face Transformers documentation: [Transformers](https://huggingface.co/docs/transformers/)
- Cosine Similarity and other metrics for NLP: [Similarity Metrics](https://medium.com/analytics-vidhya/)
- Meta’s LLaMA model research and applications: [LLaMA Paper](https://arxiv.org/abs/2302.10149)

**Note**: Make sure to check for access permissions if using restricted models from Hugging Face and set up your environment with `requirements.txt`.

---

*This README.md provides a summary of work completed so far. The fine-tuning process and deployment steps will be detailed further in the next project stage.*
