"""
Embedding module for FairHire AI.
Wraps MiniLM and DeBERTa models behind a simple interface.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class CandidateEmbedder:
    """
    Turns resume text into numerical vectors using transformer models.

    Two models available:
        - 'minilm'  : fast, 384-dim vectors, good for similarity (primary)
        - 'deberta' : slower, 768-dim vectors, used for robustness check
    """

    MODELS = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "deberta": "microsoft/deberta-v3-base",
    }

    def __init__(self, model_type="minilm", device="cpu"):
        if model_type not in self.MODELS:
            raise ValueError(f"Use 'minilm' or 'deberta', got '{model_type}'")

        self.model_type = model_type
        self.device = device
        self.hf_id = self.MODELS[model_type]

        print(f"Loading {model_type} ({self.hf_id})...")

        if model_type == "minilm":
            self.model = SentenceTransformer(self.hf_id, device=device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
            self.model = AutoModel.from_pretrained(self.hf_id).to(device)
            self.model.eval()

        print(f"Ready on {device}.\n")

    def encode(self, texts, batch_size=32):
        """Convert text(s) into normalised embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "minilm":
            return self.model.encode(
                texts, device=self.device,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )

        # DeBERTa: manual tokenize → forward pass → mean pool → normalize
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**tokens)

            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embs.append(normed.cpu().numpy())

        return np.vstack(all_embs)

    def compare(self, text_a, text_b):
        """Cosine similarity between two texts. Returns float in [-1, 1]."""
        return float(cosine_similarity(self.encode(text_a), self.encode(text_b))[0][0])

    def rank(self, query, candidates):
        """Score each candidate against a query. Returns list of floats."""
        q = self.encode(query)
        c = self.encode(candidates)
        return cosine_similarity(q, c)[0].tolist()