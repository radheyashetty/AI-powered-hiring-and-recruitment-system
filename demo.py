from src.embedder import CandidateEmbedder

def main():
    print("==================================================")
    print("       FAIRHIRE AI - MODEL TESTING DEMO           ")
    print("==================================================")

    # Initialize Embedder (Primary Model)
    embedder = CandidateEmbedder(model_type="minilm")

    # Target Job Description
    job_description = (
        "Seeking a Python Developer with experience in Machine Learning, "
        "SQL databases, and PyTorch for AI pipeline development."
    )

    # Test Candidates
    candidates = [
        "Candidate A: B.Tech CS graduate. Proficient in Python, PyTorch, Scikit-learn, SQL, and Git.",
        "Candidate B: Graphic Designer with expertise in Photoshop, Illustrator, UI/UX, and HTML.",
        "Candidate C: Backend engineer with Java, Spring Boot, MySQL, and Docker experience."
    ]

    print("\n--> Encoding Job Description and Candidates...")
    jd_embedding = embedder.encode(job_description)
    candidate_embeddings = embedder.encode(candidates)

    print("\n=== MATCHING RESULTS (Similarity to Job Description) ===")
    for i, candidate in enumerate(candidates):
        score = embedder.compare(job_description, candidate)
        print(f"\n{candidate[:40]}...")
        print(f"   Match Score: {score * 100:.2f}%")

if __name__ == "__main__":
    main()