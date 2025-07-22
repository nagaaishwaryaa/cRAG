import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return full_text

# Split into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# Main indexing pipeline
def index_pdf(pdf_path):
    print("🔍 Loading and splitting PDF...")
    raw_text = load_pdf_text(pdf_path)
    chunks = split_text(raw_text)

    print("🔗 Embedding and creating FAISS index...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, "vector_index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    print("✅ Indexing complete.")

# Call the function with your file
index_pdf("document.pdf")  # <-- Change to your filename
