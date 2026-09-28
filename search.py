import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 4


# -----------------------------
# Main search
# -----------------------------

def main():

    print("=" * 60)
    print("RAG CHATBOT - PHASE 3 SEARCH")
    print("=" * 60)

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        print("\nChromaDB collection not found.")
        print("Run ingest.py first.")
        return

    print(
        f"\nDocuments in database: "
        f"{collection.count()}"
    )

    print("\nType your question.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Question: ").strip()

        if question.lower() == "exit":
            print("\nExiting...")
            break

        if not question:
            print("Please enter a question.")
            continue

        # Convert question to embedding
        question_embedding = model.encode(
            question
        ).tolist()

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=TOP_K
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print("\n" + "=" * 60)
        print("TOP 4 MATCHING CHUNKS")
        print("=" * 60)

        for i, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1
        ):

            print(f"\n--- Result {i} ---")
            print(f"Source   : {metadata['source']}")
            print(f"Page     : {metadata['page']}")
            print(f"Distance : {distance:.4f}")
            print("\nChunk:")
            print(document)

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()