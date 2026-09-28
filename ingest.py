from pathlib import Path
import hashlib

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------
# Chunking function
# -----------------------------

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text while trying to preserve complete policy statements."""

    text = " ".join(text.split())

    if not text:
        return []

    # Split into sentences while keeping the sentence text.
    import re

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # If adding the sentence stays within the limit,
        # keep building the current chunk.
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:

            if current_chunk:
                current_chunk += " "

            current_chunk += sentence

        else:

            # Save the current chunk.
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start a new chunk with the sentence.
            current_chunk = sentence

    # Save the final chunk.
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# -----------------------------
# Find PDFs
# -----------------------------

def find_pdfs():
    """Find all PDF files inside data/."""

    pdf_files = list(DATA_DIR.rglob("*.pdf"))

    return pdf_files


# -----------------------------
# Main ingestion
# -----------------------------

def main():

    print("=" * 60)
    print("RAG CHATBOT - PHASE 3 INGESTION")
    print("=" * 60)

    # Find PDFs
    pdf_files = find_pdfs()

    if not pdf_files:
        print("\nNo PDF files found inside the data/ folder.")
        print("Add your PDFs to data/ or data/documents/")
        return

    print(f"\nFound {len(pdf_files)} PDF file(s).")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create ChromaDB client
    print("Creating ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recreate collection so ingestion starts clean
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    total_chunks = 0

    # Process every PDF
    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path}")

        try:
            reader = PdfReader(str(pdf_path))

            print(f"Pages: {len(reader.pages)}")

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if not text or not text.strip():
                    continue

                chunks = split_text(text)

                for chunk_number, chunk in enumerate(chunks):

                    documents.append(chunk)

                    metadatas.append({
                        "source": pdf_path.name,
                        "page": page_number
                    })

                    # Create a unique ID
                    raw_id = (
                        f"{pdf_path}_{page_number}_"
                        f"{chunk_number}_{chunk}"
                    )

                    chunk_id = hashlib.md5(
                        raw_id.encode("utf-8")
                    ).hexdigest()

                    ids.append(chunk_id)

                    total_chunks += 1

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    if not documents:
        print("\nNo text was extracted from the PDFs.")
        return

    # Create embeddings
    print(f"\nCreating embeddings for {len(documents)} chunks...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    # Store in ChromaDB
    print("\nStoring chunks in ChromaDB...")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)
    print(f"PDF files       : {len(pdf_files)}")
    print(f"Total chunks    : {total_chunks}")
    print(f"Chunk size      : {CHUNK_SIZE}")
    print(f"Chunk overlap   : {CHUNK_OVERLAP}")
    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"ChromaDB        : {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()