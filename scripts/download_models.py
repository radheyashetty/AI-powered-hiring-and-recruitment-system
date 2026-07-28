"""
Downloads both models so they're cached for offline use.
Run once: python scripts/download_models.py
"""

from src.embedder import CandidateEmbedder

print("Downloading models...\n")

minilm_embedder = CandidateEmbedder("minilm")
assert minilm_embedder.encode("test").shape[1] == 384
print("MiniLM OK.\n")

deberta_embedder = CandidateEmbedder("deberta")
assert deberta_embedder.encode("test").shape[1] == 768
print("DeBERTa OK.\n")

print("Both models cached. They'll load instantly next time.")
