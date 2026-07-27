"""
Downloads both models so they're cached for offline use.
Run once: python scripts/download_models.py
"""

from src.embedder import CandidateEmbedder

print("Downloading models...\n")

m = CandidateEmbedder("minilm")
assert m.encode("test").shape[1] == 384
print("MiniLM OK.\n")

d = CandidateEmbedder("deberta")
assert d.encode("test").shape[1] == 768
print("DeBERTa OK.\n")

print("Both models cached. They'll load instantly next time.")
