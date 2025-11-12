import os
import re
import pdfplumber
import concurrent.futures
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np
import time


# -------------------------------
# 1. Text Cleaning & Chunking
# -------------------------------
def clean_text(text):
    """Cleans up whitespace and newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path):
    """Extracts text from an entire PDF."""
    print(f"📘 Reading {os.path.basename(pdf_path)} ...")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)


def chunk_text(text, chunk_size=3000, overlap=300):
    """Splits text into overlapping chunks for better semantic continuity."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# -------------------------------
# 2. Model Loading (Offline Cache)
# -------------------------------
def load_model():
    """
    Loads the SentenceTransformer model.
    If already downloaded, it’ll use the local safetensors cache.
    """
    model_name = "BAAI/bge-small-en-v1.5"
    local_dir = os.path.expanduser("~/.cache/queryiq_models/bge-small-en-v1.5")

    if not os.path.exists(local_dir):
        print("⬇️  Downloading model (first time only)...")
        model = SentenceTransformer(model_name)
        model.save(local_dir)
        print(f"✅ Model cached at: {local_dir}")
    else:
        print(f"📦 Loading model from local cache: {local_dir}")
        model = SentenceTransformer(local_dir)
    return model


# -------------------------------
# 3. Embedding Generation (Concurrent)
# -------------------------------
def embed_texts(chunks, model, batch_size=8, max_workers=4):
    """Encodes text chunks into embeddings concurrently."""
    embeddings = []

    def process_batch(batch):
        return model.encode(
            batch, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            futures.append(executor.submit(process_batch, batch))
        for f in concurrent.futures.as_completed(futures):
            embeddings.extend(f.result())

    return np.array(embeddings, dtype="float32")


# -------------------------------
# 4. Vector Index (hnswlib)
# -------------------------------
def build_index(embeddings):
    """Builds a cosine similarity index for semantic search."""
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=100, M=8)
    index.add_items(embeddings)
    index.set_ef(64)
    return index


# -------------------------------
# 5. Search
# -------------------------------
def semantic_search(query, model, index, chunks, top_k=3):
    """Performs semantic search on the indexed document."""
    query_emb = model.encode([query], convert_to_numpy=True)
    labels, distances = index.knn_query(query_emb, k=top_k)
    return [(chunks[i], float(distances[0][j])) for j, i in enumerate(labels[0])]


# -------------------------------
# 6. Main
# -------------------------------
if __name__ == "__main__":
    start_time = time.time()
    pdf_path = "example.pdf"

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=1200, overlap=150)
    print(f"✅ Created {len(chunks)} chunks from {len(text.split())} words.")

    model = load_model()

    print("⚙️ Generating embeddings...")
    embeddings = embed_texts(chunks, model)
    print(
        f"✅ Done in {time.time() - start_time:.2f}s with {embeddings.shape[0]} vectors."
    )

    index = build_index(embeddings)
    print(f"✅ Index built ({index.get_current_count()} vectors).")

    for i in range(10):
        query = str(input("Enter your query: "))
        results = semantic_search(query, model, index, chunks)

        for idx, (text, score) in enumerate(results):
            print(f"\nResult {idx + 1} (score: {score:.4f}):\n{text[:400]}...")
