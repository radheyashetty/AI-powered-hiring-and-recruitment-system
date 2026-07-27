"""
Tests for the embedding module.
Run: python -m pytest tests/ -v
"""

import numpy as np
import pytest
from src.embedder import CandidateEmbedder


# Load models once for all tests (saves time)
@pytest.fixture(scope="module")
def minilm():
    return CandidateEmbedder("minilm")

@pytest.fixture(scope="module")
def deberta():
    return CandidateEmbedder("deberta")


# -- MiniLM basic tests --

def test_minilm_single(minilm):
    out = minilm.encode("Python developer.")
    assert out.shape == (1, 384)

def test_minilm_batch(minilm):
    out = minilm.encode(["Resume A.", "Resume B.", "Resume C."])
    assert out.shape == (3, 384)

def test_minilm_normalised(minilm):
    embs = minilm.encode(["Some text.", "Other text."])
    norms = np.linalg.norm(embs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# -- DeBERTa basic tests --

def test_deberta_single(deberta):
    out = deberta.encode("Python developer.")
    assert out.shape == (1, 768)

def test_deberta_batch(deberta):
    out = deberta.encode(["Resume A.", "Resume B."])
    assert out.shape == (2, 768)

def test_deberta_normalised(deberta):
    embs = deberta.encode(["Some text."])
    norm = np.linalg.norm(embs[0])
    assert abs(norm - 1.0) < 1e-4


# -- Semantic tests --

def test_similar_resumes_score_higher(minilm):
    """Two ML resumes should be more similar than ML vs art."""
    ml_a = "Python developer with PyTorch and scikit-learn."
    ml_b = "ML engineer experienced in Python and TensorFlow."
    art  = "Oil painter specialising in abstract landscapes."

    assert minilm.compare(ml_a, ml_b) > minilm.compare(ml_a, art)

def test_identical_text_scores_one(minilm):
    text = "B.Tech CS from IIT Delhi."
    assert minilm.compare(text, text) > 0.999

def test_rank_returns_correct_count(minilm):
    scores = minilm.rank("Data scientist needed.", ["DS resume.", "Chef.", "Pilot."])
    assert len(scores) == 3

def test_rank_picks_right_candidate(minilm):
    scores = minilm.rank(
        "Hiring data scientist with Python and statistics.",
        [
            "Chef with 10 years of restaurant experience.",
            "Data scientist with Python, R, and ML skills.",
            "Truck driver with commercial licence.",
        ]
    )
    assert np.argmax(scores) == 1  # data scientist should win


# -- Team complementarity --

def test_different_skills_are_more_complementary(minilm):
    """Someone with different skills should have higher complement score."""
    from sklearn.metrics.pairwise import cosine_similarity

    team = ["Python, PyTorch, ML, SQL.", "Python, TensorFlow, DL, AWS."]
    similar = "Python ML engineer with PyTorch and SQL."
    different = "Graphic designer with Photoshop and Illustrator."

    centroid = np.mean(minilm.encode(team), axis=0, keepdims=True)
    sim_to_team = cosine_similarity(minilm.encode(similar), centroid)[0][0]
    diff_to_team = cosine_similarity(minilm.encode(different), centroid)[0][0]

    # similar candidate should be CLOSER to centroid
    assert sim_to_team > diff_to_team


# -- Error handling --

def test_bad_model_type():
    with pytest.raises(ValueError):
        CandidateEmbedder("gpt4")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
