# 📚 QueryIQ

**QueryIQ** is an advanced RAG (Retrieval-Augmented Generation) application that allows users to chat with their PDF documents. It uses local embeddings for fast semantic search and Google's Gemini API to generate natural language answers.

## 🚀 Features

- **Multi-PDF Support**: Upload and process multiple PDF files simultaneously.
- **AI-Powered Chat**: Ask questions in natural language and get answers based *only* on your documents.
- **Source Citations**: Every answer includes references to the specific file and page number.
- **Privacy-Focused Search**: Uses local embeddings (`BAAI/bge-small-en-v1.5`) and local vector storage (`hnswlib`).
- **Premium UI**: Built with Streamlit and styled for a modern, dark-mode experience.

## 🛠️ Installation

This project uses `uv` for dependency management, but standard `pip` works too.

### Prerequisites
- Python 3.13+
- A Google Gemini API Key (Get it [here](https://aistudio.google.com/))

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/QueryIQ.git
   cd QueryIQ
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (Recommended)
   uv sync

   # OR using pip
   pip install .
   ```

## 🏃‍♂️ Usage

1. **Run the Streamlit App**:
   ```bash
   streamlit run main.py
   ```

2. **Configure**:
   - Open the app in your browser (usually `http://localhost:8501`).
   - Enter your **Google Gemini API Key** in the sidebar.

3. **Chat**:
   - Upload your PDF documents.
   - Click **Process Documents**.
   - Start asking questions!

## 🏗️ Architecture

- **Frontend**: Streamlit
- **Embeddings**: `sentence-transformers` (BAAI/bge-small-en-v1.5)
- **Vector Store**: `hnswlib` (Approximate Nearest Neighbor Search)
- **LLM**: Google Gemini Pro (via `google-generativeai`)
- **PDF Processing**: `pdfplumber`

## 📄 License

MIT License
