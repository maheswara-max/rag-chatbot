import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Reuse the existing Phase 7 RAG functions
from chat import search_documents, generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 8
RERANK_TOP_K = 4


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 HR RAG Chatbot")

st.caption(
    "Ask questions about HR policies, employee benefits, leave, "
    "attendance, work from home, resignation, and other HR topics."
)

st.info(
    "Example questions: "
    "What is the leave policy? | "
    "What is the work-from-home policy? | "
    "What are the employee benefits?"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# LOAD RERANKER MODEL
# ============================================================

@st.cache_resource
def load_reranker():

    return CrossEncoder(
        RERANKER_MODEL
    )


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# ============================================================
# LOAD RAG RESOURCES
# ============================================================

try:

    with st.spinner("Loading RAG system..."):

        embedding_model = load_embedding_model()

        reranker = load_reranker()

        collection = load_collection()

except Exception as e:

    st.error("❌ Failed to load the RAG system.")

    st.error(str(e))

    st.info(
        "Make sure the ChromaDB collection exists "
        "and the required models are available."
    )

    st.stop()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Display sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**📄 {source['source']}**  \n"
                        f"Page: **{source['page']}**"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about HR policies..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Save and display user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        # ----------------------------------------------------
        # Search documents
        # ----------------------------------------------------

        with st.spinner(
            "Searching HR policy documents..."
        ):

            try:

                chunks = search_documents(
                    question,
                    embedding_model,
                    collection,
                    reranker
                )

            except Exception as e:

                st.error(
                    "❌ Error while searching documents."
                )

                st.error(str(e))

                st.stop()


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    chunks
                )

            except Exception as e:

                st.error(
                    "❌ Error while generating answer."
                )

                st.error(str(e))

                st.stop()


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # Prepare unique sources
        # ----------------------------------------------------

        sources = []

        seen = set()

        for chunk in chunks:

            source = chunk.get(
                "source",
                "Unknown"
            )

            page = chunk.get(
                "page",
                "Unknown"
            )

            source_key = (
                source,
                page
            )

            if source_key not in seen:

                seen.add(
                    source_key
                )

                sources.append(
                    {
                        "source": source,
                        "page": page
                    }
                )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.markdown(
                        f"**📄 {source['source']}**  \n"
                        f"Page: **{source['page']}**"
                    )

        else:

            st.info(
                "No sources found."
            )


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ About")

    st.write(
        "This chatbot uses Retrieval-Augmented Generation "
        "(RAG) to answer questions from the HR document "
        "knowledge base."
    )

    st.divider()

    st.write("**Embedding Model**")

    st.code(
        EMBEDDING_MODEL
    )

    st.write("**Reranker Model**")

    st.code(
        RERANKER_MODEL
    )

    st.write("**Vector Database**")

    st.code(
        "ChromaDB"
    )

    st.write("**Retrieval**")

    st.code(
        "Top 8 → Reranked → Top 4"
    )

    st.write("**LLM**")

    st.code(
        "Groq / GPT-OSS-20B"
    )

    st.divider()

    st.write(
        "**Supported topics**"
    )

    st.write(
        """
        • Leave Policy  
        • Attendance Policy  
        • Work From Home  
        • Employee Benefits  
        • Health Insurance  
        • Parental Leave  
        • Recruitment  
        • Resignation  
        • Travel Expenses  
        • Code of Conduct
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()