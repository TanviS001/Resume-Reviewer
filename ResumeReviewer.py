import pdfplumber
import numpy as np
from numpy import dot
from numpy.linalg import norm
from huggingface_hub import InferenceClient
from groq import Groq

HF_API_KEY = ""
GROQ_API_KEY = ""
RESUME_FILE = "resume.pdf"
JOB_DESC_FILE = "JD.txt"

def load_resume(filepath):
    """Extract text from PDF resume."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def load_job_description(filepath):
    """Read job description text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def compute_embeddings(texts, hf_client):
    """Generate vector embeddings using Hugging Face sentence-transformer."""
    embeddings = hf_client.feature_extraction(
        texts,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    return np.array(embeddings, dtype="float32")

def compute_similarity(resume_text, job_desc, hf_client):
    """Compute cosine similarity between resume and job description."""
    v1, v2 = compute_embeddings([resume_text, job_desc], hf_client)
    return dot(v1, v2) / (norm(v1) * norm(v2))

def generate_review(groq_client, resume_text, job_desc, score):
    """Generate structured resume feedback using Groq LLM."""
    prompt = f"""
Job Description:
{job_desc}

Resume:
{resume_text}

Resume-Job Match Score: {score:.2f}

Provide a detailed review with:
1. Strengths of this resume for the given job.
2. Weaknesses or gaps compared to the job description.
3. Overall verdict on hiring potential.
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    hf_client = InferenceClient(token=HF_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)

    resume_text = load_resume(RESUME_FILE)
    job_desc = load_job_description(JOB_DESC_FILE)

    print("Evaluating resume vs job description...")
    score = compute_similarity(resume_text, job_desc, hf_client)
    print(f"Resume-Job Match Score: {score:.2f}")

    print("\nGenerating resume review...")
    review = generate_review(groq_client, resume_text, job_desc, score)
    print("\nResume Review Report:\n")
    print(review)
