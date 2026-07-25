import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

print("==================================================")
print("              FAIRHIRE AI - CPU TEST              ")
print("==================================================")

# Sample resumes for testing
resumes = [
    "B.Tech Computer Science student from IIT Delhi. Skills: Python, Machine Learning, PyTorch, SQL, Data Structures.",
    "B.Tech Information Technology student from Tier-3 College. Skills: Java, HTML, CSS, JavaScript, MySQL."
]

# ---------------------------------------------------------
# 1. PRIMARY MODEL: all-MiniLM-L6-v2 (384 dims)
# ---------------------------------------------------------
print("\n1. Testing Primary Model (all-MiniLM-L6-v2)...")
start = time.time()
primary_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print(f"   Loaded in {time.time() - start:.2f} seconds.")

# Generate embeddings
embeddings_minilm = primary_model.encode(resumes, device="cpu", normalize_embeddings=True)
print(f"   Embedding Matrix Shape: {embeddings_minilm.shape}")  # (2, 384)

sim_minilm = cosine_similarity([embeddings_minilm[0]], [embeddings_minilm[1]])[0][0]
print(f"   Cosine Similarity (Resume 1 vs 2): {sim_minilm:.4f}")


# ---------------------------------------------------------
# 2. SECONDARY MODEL: deberta-v3-base (768 dims, Frozen)
# ---------------------------------------------------------
print("\n2. Testing Secondary Model (deberta-v3-base)...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
secondary_model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
secondary_model.eval()  # Freeze weights
print(f"   Loaded in {time.time() - start:.2f} seconds.")

def get_deberta_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = secondary_model(**inputs)
    token_embs = outputs.last_hidden_state  # (batch, seq_len, 768)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    mean_embs = (token_embs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(mean_embs, p=2, dim=1).numpy()

embeddings_deberta = get_deberta_embeddings(resumes)
print(f"   Embedding Matrix Shape: {embeddings_deberta.shape}")  # (2, 768)

sim_deberta = cosine_similarity([embeddings_deberta[0]], [embeddings_deberta[1]])[0][0]
print(f"   Cosine Similarity (Resume 1 vs 2): {sim_deberta:.4f}")

print("\n==================================================")
print(" SUCCESS: Both models downloaded & verified on CPU!")
print("==================================================")