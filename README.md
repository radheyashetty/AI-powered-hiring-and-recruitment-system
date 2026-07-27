# FairHire AI

An explainable and fair AI hiring pipeline that investigates how demographic debiasing interacts with team-skill complementarity scoring.

## What this does

- Encodes candidate resumes into vector embeddings
- Scores candidates for job fit and team complementarity
- Applies fairness debiasing (Fairlearn)
- Generates per-candidate SHAP explanations
- Tests whether debiasing changes who ranks highest on team diversity

## Quick start

```bash
# setup
python -m venv venv
.\venv\Scripts\Activate.ps1       # windows
pip install -r requirements.txt

# download models (~500MB total, cached after first run)
python scripts/download_models.py

# run demo
python demo.py

# run tests
python -m pytest tests/ -v
```

## Project structure

```
src/
  embedder.py         # embedding wrapper for MiniLM + DeBERTa
scripts/
  download_models.py  # downloads and caches both models
tests/
  test_embedder.py    # unit tests
demo.py               # interactive demo with sample candidates
```

## Models used

| Model | Role | Dimensions | Size |
|---|---|---|---|
| all-MiniLM-L6-v2 | Primary (similarity-trained) | 384 | 80 MB |
| DeBERTa-v3-base | Secondary (robustness check) | 768 | ~400 MB |

## Tech stack

Python, sentence-transformers, transformers, fairlearn, scikit-learn, shap
