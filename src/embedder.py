import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class CandidateEmbedder:
    """
    A wrapper class to extract candidate embeddings using primary (MiniLM)
    and secondary (DeBERTa-v3) transformer models.
    """

    def __init__(self, model_type: str = "minilm", device: str = "cpu"):
        self.model_type = model_type.lower()
        self.device = device

        if self.model_type == "minilm":
            print(f"[Info] Loading primary model: all-MiniLM-L6-v2 on {self.device}...")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        elif self.model_type == "deberta":
            print(f"[Info] Loading secondary model: deberta-v3-base on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(self.device)
            self.model.eval()
        else:
            raise ValueError("Unsupported model_type. Choose 'minilm' or 'deberta'.")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generates L2-normalized embeddings for input text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "minilm":
            return self.model.encode(texts, device=self.device, normalize_embeddings=True)

        elif self.model_type == "deberta":
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling over token dimension
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # L2 Normalize
            normalized = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            return normalized.cpu().numpy()

    def compare(self, text_a: str, text_b: str) -> float:
        """Utility method to compute cosine similarity between two texts."""
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        return float(cosine_similarity(emb_a, emb_b)[0][0])