"""
Demo script — shows what the embedding model can do.
Run: python demo.py
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedder import CandidateEmbedder


# -- Sample data (Indian engineering student profiles) --

job = (
    "Hiring a Machine Learning Engineer. "
    "Skills needed: Python, PyTorch, scikit-learn, SQL, REST APIs. "
    "NLP or computer vision experience is a plus."
)

candidates = {
    "Arjun (IIT Delhi, CS)":
        "B.Tech CS from IIT Delhi. Interned at Google Research on NLP. "
        "Skills: Python, PyTorch, TensorFlow, SQL, Docker.",

    "Priya (NIT Trichy, ECE)":
        "B.Tech ECE from NIT Trichy. Interned at Texas Instruments. "
        "Skills: C, C++, MATLAB, Embedded Linux, VHDL.",

    "Rohan (Tier-3, IT)":
        "B.Tech IT from a regional college. Freelance web dev. "
        "Skills: HTML, CSS, JavaScript, React, Node.js, MySQL.",

    "Sneha (IIIT Hyd, AI)":
        "B.Tech AI from IIIT Hyderabad. Research intern at MSR on CV. "
        "Skills: Python, PyTorch, OpenCV, scikit-learn, AWS.",

    "Karthik (VIT, CS)":
        "B.Tech CS from VIT. Interned at Infosys on backend APIs. "
        "Skills: Java, Spring Boot, Python, SQL, MongoDB.",
}

team = [
    "Senior ML Engineer: Python, TensorFlow, Keras, SQL, MLflow.",
    "Data Engineer: Python, PySpark, Airflow, SQL, AWS.",
    "Backend Dev: Java, Spring Boot, PostgreSQL, Docker.",
]


def test_job_matching(emb):
    """Which candidate fits the job best?"""
    print("\n--- Test 1: Job Matching ---")
    print(f"Job: {job[:60]}...\n")

    names = list(candidates.keys())
    scores = emb.rank(job, list(candidates.values()))

    ranked = sorted(zip(names, scores), key=lambda x: -x[1])
    for i, (name, score) in enumerate(ranked, 1):
        print(f"  {i}. {name:<28} {score*100:.1f}%")


def test_similarity(emb):
    """Are similar profiles detected as similar?"""
    print("\n--- Test 2: Candidate Similarity ---\n")

    names = list(candidates.keys())
    texts = list(candidates.values())

    pairs = [(0, 3), (0, 2), (1, 2)]  # Arjun-Sneha, Arjun-Rohan, Priya-Rohan
    for i, j in pairs:
        score = emb.compare(texts[i], texts[j])
        print(f"  {names[i]:<25} vs {names[j]:<25} -> {score*100:.1f}%")


def test_complementarity(emb):
    """Who fills gaps in the existing team?"""
    print("\n--- Test 3: Team Complementarity ---")
    print("  (Higher = brings more new skills)\n")

    team_embs = emb.encode(team)
    centroid = np.mean(team_embs, axis=0, keepdims=True)

    names = list(candidates.keys())
    cand_embs = emb.encode(list(candidates.values()))

    sims = cosine_similarity(cand_embs, centroid).flatten()
    comp_scores = 1.0 - sims

    ranked = sorted(zip(names, comp_scores), key=lambda x: -x[1])
    for i, (name, score) in enumerate(ranked, 1):
        print(f"  {i}. {name:<28} complement: {score*100:.1f}%")


def test_model_comparison():
    """Do MiniLM and DeBERTa agree on who's the best candidate?"""
    print("\n--- Test 4: MiniLM vs DeBERTa ---\n")

    m = CandidateEmbedder("minilm")
    d = CandidateEmbedder("deberta")

    names = list(candidates.keys())
    texts = list(candidates.values())

    m_scores = m.rank(job, texts)
    d_scores = d.rank(job, texts)

    print(f"  {'Candidate':<28} {'MiniLM':>8} {'DeBERTa':>8}")
    print(f"  {'-'*46}")
    for name, ms, ds in zip(names, m_scores, d_scores):
        print(f"  {name:<28} {ms*100:>6.1f}% {ds*100:>6.1f}%")

    top_m = names[np.argmax(m_scores)]
    top_d = names[np.argmax(d_scores)]
    print(f"\n  MiniLM  picks: {top_m}")
    print(f"  DeBERTa picks: {top_d}")


if __name__ == "__main__":
    print("FairHire AI — Demo\n")

    emb = CandidateEmbedder("minilm")

    test_job_matching(emb)
    test_similarity(emb)
    test_complementarity(emb)

    run_t4 = input("\nRun DeBERTa comparison too? (loads ~400MB) [y/N]: ").strip().lower()
    if run_t4 == "y":
        test_model_comparison()

    print("\nDone.")