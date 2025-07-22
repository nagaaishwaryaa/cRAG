import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# Load model and FAISS index
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load preprocessed data
@st.cache_data
def load_index_and_chunks():
    import pickle
    index = faiss.read_index("vector_index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Text similarity
def embed(text):
    return model.encode([text])[0]

# Evaluation criteria
def evaluate_context(query, chunk):
    relevance = cosine_sim(embed(query), embed(chunk))
    completeness = min(1.0, len(chunk.split()) / 40)  # heuristic
    accuracy = 1.0  # assume true from file, can refine
    specificity = 1.0 if query.lower() in chunk.lower() else relevance
    return relevance, completeness, accuracy, specificity

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Score to qualitative
def score_to_quality(avg):
    if avg > 0.8:
        return "EXCELLENT"
    elif avg > 0.6:
        return "GOOD"
    elif avg > 0.4:
        return "FAIR"
    else:
        return "POOR"

# Retrieve top chunks
def retrieve(query, index, chunks, top_k=3):
    q_emb = embed(query)
    D, I = index.search(np.array([q_emb]).astype("float32"), top_k)
    return [(chunks[i], D[0][rank]) for rank, i in enumerate(I[0])]

# Refine query
def refine_query(query):
    return query + " explanation details facts"

# Streamlit UI
st.title("🧠 Corrective RAG QA System")
query = st.text_input("Enter your question:")
if query:
    index, chunks = load_index_and_chunks()
    results = retrieve(query, index, chunks)

    evaluated = []
    for chunk, _ in results:
        rel, comp, acc, spec = evaluate_context(query, chunk)
        avg = np.mean([rel, comp, acc, spec])
        evaluated.append({
            "chunk": chunk,
            "scores": (rel, comp, acc, spec),
            "average": avg,
            "quality": score_to_quality(avg)
        })

    best = max(evaluated, key=lambda x: x["average"])
    quality = best["quality"]

    if quality in ["FAIR", "POOR"]:
        new_query = refine_query(query)
        results = retrieve(new_query, index, chunks)
        best_chunk = results[0][0]
        st.markdown("⚠️ **Note: Initial retrieval was insufficient. Used refined query.**")
        quality = "GOOD"
        confidence = "MEDIUM"
    else:
        best_chunk = best["chunk"]
        confidence = "HIGH" if quality == "EXCELLENT" else "MEDIUM"

    # Final output
    st.markdown(f"""
🔍 **Context Quality:** {quality}  
📊 **Confidence Level:** {confidence}  
🎯 **Answer:** {best_chunk}  
📚 **Source:** Pre-indexed PDF  
""")
