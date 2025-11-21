import os
import re
import pdfplumber
import concurrent.futures
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np
import time
import google.generativeai as genai
from transformers import pipeline

# -------------------------------
# 1. Text Cleaning & Chunking
# -------------------------------
def clean_text(text):
    """Cleans up whitespace and newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdfs(pdf_paths):
    """
    Extracts text from a list of PDF files.
    Returns a list of dictionaries: [{'text': ..., 'source': ..., 'page': ...}, ...]
    """
    documents = []
    for pdf_path in pdf_paths:
        print(f"📘 Reading {os.path.basename(pdf_path)} ...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = clean_text(page_text)
                        if cleaned_text:
                            documents.append({
                                "text": cleaned_text,
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1
                            })
        except Exception as e:
            print(f"❌ Error reading {pdf_path}: {e}")
    return documents


def chunk_text(documents, chunk_size=1000, overlap=200):
    """
    Splits text into overlapping chunks while preserving metadata.
    Input: List of dicts {'text': ..., 'source': ..., 'page': ...}
    Output: List of dicts {'text': ..., 'source': ..., 'page': ...}
    """
    chunks = []
    for doc in documents:
        text = doc["text"]
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"]
            })
            
            start += chunk_size - overlap
            
    return chunks


# -------------------------------
# 2. Model Loading (Offline Cache)
# -------------------------------
def load_model():
    """
    Loads the SentenceTransformer model for embeddings.
    """
    model_name = "BAAI/bge-small-en-v1.5"
    local_dir = os.path.expanduser("~/.cache/queryiq_models/bge-small-en-v1.5")

    if not os.path.exists(local_dir):
        print("⬇️  Downloading embedding model (first time only)...")
        model = SentenceTransformer(model_name)
        model.save(local_dir)
        print(f"✅ Embedding model cached at: {local_dir}")
    else:
        print(f"📦 Loading embedding model from local cache: {local_dir}")
        model = SentenceTransformer(local_dir)
    return model


def load_local_llm():
    """
    Loads the local LaMini model for text generation.
    """
    model_id = "MBZUAI/LaMini-Flan-T5-248M"
    local_dir = os.path.expanduser("~/.cache/queryiq_models/lamini-flan-t5-248m")
    
    print("⚙️ Loading Local LLM (LaMini)...")
    # We rely on transformers caching, but we can specify a cache dir
    llm_pipeline = pipeline(
        "text2text-generation",
        model=model_id,
        model_kwargs={"cache_dir": local_dir},
        max_length=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    print("✅ Local LLM loaded.")
    return llm_pipeline


# -------------------------------
# 3. Embedding Generation (Concurrent)
# -------------------------------
def embed_texts(chunks, model, batch_size=8, max_workers=4):
    """Encodes text chunks into embeddings concurrently."""
    text_list = [chunk["text"] for chunk in chunks]
    embeddings = []

    def process_batch(batch):
        return model.encode(
            batch, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            futures.append(executor.submit(process_batch, batch))
        for f in concurrent.futures.as_completed(futures):
            embeddings.extend(f.result())

    return np.array(embeddings, dtype="float32")


# -------------------------------
# 4. Vector Index (hnswlib)
# -------------------------------
def build_index(embeddings):
    """Builds a cosine similarity index for semantic search."""
    if len(embeddings) == 0:
        return None
        
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(embeddings), ef_construction=100, M=16)
    index.add_items(embeddings)
    index.set_ef(64)
    return index


# -------------------------------
# 5. Search & Generation
# -------------------------------
def semantic_search(query, model, index, chunks, top_k=3):
    """Performs semantic search on the indexed document."""
    if index is None or not chunks:
        return []
        
    query_emb = model.encode([query], convert_to_numpy=True)
    labels, distances = index.knn_query(query_emb, k=min(top_k, len(chunks)))
    
    results = []
    for j, i in enumerate(labels[0]):
        results.append({
            "chunk": chunks[i],
            "score": float(distances[0][j])
        })
    return results


def generate_answer_gemini(query, context_text, api_key):
    """Generates answer using Google Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
If the answer is not in the context, say "I cannot find the answer in the provided documents."

Context:
{context_text}

Question: {query}

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer with Gemini: {str(e)}"


def generate_answer_local(query, context_text, llm_pipeline):
    """Generates answer using local LaMini model."""
    try:
        # Truncate context to avoid token limit (512 tokens ~ 2000 chars)
        # We leave some room for the query and prompt structure
        max_context_chars = 1500
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars] + "..."
            
        prompt = f"Answer the question based on the context below.\n\nContext: {context_text}\n\nQuestion: {query}\n\nAnswer:"
        
        result = llm_pipeline(prompt)
        return result[0]['generated_text']
    except Exception as e:
        return f"❌ Error generating answer with Local AI: {str(e)}"


def generate_answer(query, context_results, model_type="high", api_key=None, local_llm=None):
    """
    Router function to generate answer based on selected model.
    model_type: "high" (Gemini) or "low" (Local)
    """
    # Construct Context Text
    context_text = "\n\n".join([
        f"Source ({r['chunk']['source']}, Page {r['chunk']['page']}):\n{r['chunk']['text']}" 
        for r in context_results
    ])

    if model_type == "high":
        if not api_key:
            return "⚠️ API Key missing. Please provide a Google Gemini API key."
        return generate_answer_gemini(query, context_text, api_key)
    
    elif model_type == "low":
        if not local_llm:
            return "⚠️ Local model not loaded."
        return generate_answer_local(query, context_text, local_llm)
    
    else:
        return "⚠️ Invalid model type selected."

