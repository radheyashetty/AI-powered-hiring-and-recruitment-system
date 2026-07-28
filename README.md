# FairHire AI

An explainable and fair AI hiring pipeline that investigates how demographic debiasing interacts with team-skill complementarity scoring.

## Overview

FairHire AI is a machine learning project designed to process candidate resumes and score them based on job fit and team complementarity. It focuses on maintaining fairness during the hiring process by analyzing how debiasing strategies (such as those provided by Fairlearn) affect the evaluation of team diversity and candidate ranking.

## Features

- **Resume Embedding:** Converts candidate resumes into numerical vectors using Transformer models.
- **Scoring & Ranking:** Calculates similarity between resumes and job descriptions using cosine similarity, ranking candidates accordingly.
- **Team Complementarity:** Identifies candidates who fill gaps in an existing team rather than duplicating current skills.
- **Demographic Debiasing:** Uses `fairlearn` to assess and mitigate demographic biases in hiring.
- **Explainability:** Employs SHAP to provide per-candidate explanations of why they scored highly or poorly.
- **Model Comparison:** Evaluates different Transformer architectures (MiniLM vs DeBERTa) to ensure robustness.

## Repository Structure

```
├── data/
│   └── Capstone Filteration.xlsx  # Dataset used for candidate filtering
├── docs/
│   ├── FairHire_AI_PRD.docx       # Product Requirements Document
│   ├── FairHire_AI_PRD_v3.docx    # Updated Product Requirements Document
│   ├── prd.md                     # Markdown version of PRD
│   └── prd2.md                    # Alternative Markdown PRD
├── scripts/
│   └── download_models.py         # Script to download and cache transformer models
├── src/
│   └── embedder.py                # Wrapper module for embedding models (MiniLM & DeBERTa)
├── tests/
│   └── test_embedder.py           # Unit tests for the embedding module
├── demo.py                        # Interactive demonstration script
├── test.py                        # Simple test script for verifying model inference
├── requirements.txt               # Required Python dependencies
└── README.md                      # Project documentation
```

## Installation

To set up the project locally, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Downloading Models

To ensure the models are available for offline use and to speed up execution, run the download script before using the project:

```bash
# Download models (~500MB total, cached after first run)
python scripts/download_models.py
```

## Required Dependencies

- `torch`: Deep learning framework
- `numpy`: Numerical operations
- `sentence-transformers` & `transformers`: NLP models for embeddings
- `scikit-learn`: Metrics and similarity computations
- `pytest`: Testing framework
- `fairlearn`: Fairness and debiasing tools
- `shap`: Explainability tools

See `requirements.txt` for the full list of minimal dependencies.

## Usage

### Running the Demo

The `demo.py` script provides an interactive look at the system's capabilities using sample candidate data.

```bash
python demo.py
```

### Running the Test Script

A simple script is included to quickly verify that the models load correctly and inference functions as expected.

```bash
python test.py
```

### Expected Output for Test Script

```
==================================================
              FAIRHIRE AI - CPU TEST
==================================================

1. Testing Primary Model (all-MiniLM-L6-v2)...
   Loaded in X.XX seconds.
   Embedding Matrix Shape: (2, 384)
   Cosine Similarity (Resume 1 vs 2): 0.XXXX

2. Testing Secondary Model (deberta-v3-base)...
   Loaded in X.XX seconds.
   Embedding Matrix Shape: (2, 768)
   Cosine Similarity (Resume 1 vs 2): 0.XXXX

==================================================
 SUCCESS: Both models downloaded & verified on CPU!
==================================================
```

### Running Unit Tests

To run the full suite of unit tests on the embedder module:

```bash
python -m pytest tests/ -v
```

## Future Improvements

- Integrate a frontend web application for easier resume uploading.
- Expand the dataset to include a wider variety of job domains.
- Allow fine-tuning of the embeddings on domain-specific resumes.
- Further refine the fairness evaluation metrics to support multi-class demographic features.
