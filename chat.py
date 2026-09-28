import os

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# Configuration
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 8
RERANK_TOP_K = 4

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Add it to your .env file."
    )


# ============================================================
# Initialize Groq
# ============================================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# Search + Rerank
# ============================================================

def search_documents(
    question,
    model,
    collection,
    reranker
):
    """
    Retrieve candidate chunks from ChromaDB
    and rerank them using a CrossEncoder.
    """

    question_embedding = model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        candidates.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance
        })

    if not candidates:
        return []

    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    pairs = [
        [question, candidate["text"]]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(
        candidates,
        scores
    ):
        candidate["rerank_score"] = float(score)

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    top_chunks = candidates[:RERANK_TOP_K]

    return top_chunks


# ============================================================
# Generate answer
# ============================================================

def generate_answer(question, chunks):
    """
    Generate an answer using only retrieved documents.
    """

    if not chunks:
        return "I don't know based on the documents."

    context_parts = []

    for chunk in chunks:

        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    prompt = f"""
You are an HR policy document question-answering assistant.

Use ONLY the information contained in the CONTEXT.

Rules:

1. Read all retrieved context carefully.

2. If the answer is explicitly stated in the context,
   answer the user's question directly.

3. Preserve important details such as:
   - numbers
   - dates
   - time periods
   - conditions
   - exceptions
   - eligibility requirements

4. Do not use outside knowledge.

5. Do not invent or guess information.

6. Do not combine unrelated policies.

7. Pay attention to the SOURCE of each piece of information.

8. If multiple documents contain different rules for the same topic,
   do not merge those rules into one answer.

9. If the user explicitly names a policy, organization, or document,
   use information from that relevant source.

10. If the question does not identify a policy and the retrieved
    documents contain clearly conflicting rules, do not choose
    one arbitrarily. State that the documents contain different
    rules and ask the user to specify which policy they mean.

11. If the context genuinely does not contain the answer, respond:

I don't know based on the documents.

12. Keep the final answer concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise HR policy assistant. "
                    "Answer only from the supplied document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Print sources
# ============================================================

def print_sources(chunks):
    """
    Print unique source files and page numbers.
    """

    if not chunks:
        return

    print("\nSources:")
    print("-" * 40)

    seen = set()

    for chunk in chunks:

        source = chunk["source"]
        page = chunk["page"]

        source_key = (
            source,
            page
        )

        if source_key not in seen:

            print(
                f"- {source}, page {page}"
            )

            seen.add(source_key)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("RAG CHATBOT - PHASE 7")
    print("=" * 60)

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Loading reranker model...")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print("Connecting to ChromaDB...")

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
        f"Documents in database: "
        f"{collection.count()}"
    )

    print("\n" + "=" * 60)
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        question = input("\nYou: ").strip()

        # Empty question
        if not question:

            print(
                "Please enter a question."
            )

            continue

        # Exit
        if question.lower() == "exit":

            print("\nGoodbye!")

            break

        # Retrieve and rerank
        chunks = search_documents(
            question,
            model,
            collection,
            reranker
        )

        # Generate answer
        answer = generate_answer(
            question,
            chunks
        )

        print("\nAssistant:")
        print(answer)

        # Show sources
        print_sources(chunks)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()