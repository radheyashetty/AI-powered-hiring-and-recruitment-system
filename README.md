# FairHire AI

## Overview
FairHire AI is a machine learning project designed to evaluate candidate resumes against job descriptions fairly and accurately. It leverages modern NLP techniques and Hugging Face transformer models to generate embeddings and compute cosine similarities, helping to identify the best candidates based on objective metrics.

## Features
- **Hugging Face Integration**: Uses `sentence-transformers/all-MiniLM-L6-v2` as the primary model and `microsoft/deberta-v3-base` as a secondary model for robust text embeddings.
- **Candidate Embedding**: Generates L2-normalized embeddings for job descriptions and candidate resumes.
- **Similarity Scoring**: Computes cosine similarity to score and rank candidates against job requirements.
- **CPU & GPU Support**: Configurable to run on either CPU or GPU environments.

## Repository Structure
```
.
├── data/                  # Dataset files (e.g., Capstone Filteration.xlsx)
├── docs/                  # Documentation and PRD files
├── src/
│   └── embedder.py        # Core logic for loading models and generating embeddings
├── demo.py                # Demonstration script for matching candidates to a job description
├── test.py                # Script to verify model loading and basic inference
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Test Script
To verify that the Hugging Face models load correctly and can process text, run the test script:

```bash
python test.py
```

### Expected Output
The test script should load both the primary (MiniLM) and secondary (DeBERTa) models, compute embeddings for sample resumes, and output the matrix shapes and cosine similarity scores. It will conclude with a success message if no errors occur.

## How to Run the Demo
To see the system match candidates against a job description, run the demo script:

```bash
python demo.py
```

## Future Improvements
- Implement a web interface or API to allow interactive candidate screening.
- Expand the model pipeline to include bias detection and mitigation steps.
- Fine-tune models on domain-specific HR datasets to improve matching accuracy.
- Enhance the repository with automated tests (e.g., using `pytest`) and continuous integration (CI) workflows.
