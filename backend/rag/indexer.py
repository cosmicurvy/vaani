from __future__ import annotations

import json
import pickle
import re
import unicodedata
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import settings

# Configuration (resolved relative to the repository root)
DATA_DIR = settings.DATA_DIR / "policies"
FAQ_PATH = settings.DATA_DIR / "faq.json"
QDRANT_PATH = settings.QDRANT_PATH
BM25_PATH = settings.BM25_PATH

COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
EMBEDDING_DEVICE = settings.EMBEDDING_DEVICE
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts while preserving paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")  # Remove null characters
    # Join words hyphenated across line breaks
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    # Normalize multiple blank lines into a double newline (paragraph break)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Convert single line breaks inside a paragraph into spaces
    text = re.sub(r"(?<!\n)[ \t]*\n[ \t]*(?!\n)", " ", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_pdfs(data_dir: Path | str = DATA_DIR) -> list[Document]:
    """Load and clean policy PDFs from a directory."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {data_dir}")

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=False)

    documents = loader.load()
    cleaned_documents: list[Document] = []

    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)
        if not doc.page_content:
            continue

        # Preserve page metadata and store a portable source filename.
        original_source = doc.metadata.get("source", "")
        doc.metadata["source"] = Path(original_source).name
        doc.metadata["source_type"] = "policy"
        cleaned_documents.append(doc)

    print(f"Loaded {len(cleaned_documents)} PDF pages.")
    return cleaned_documents


def load_faqs(faq_path: Path | str = FAQ_PATH) -> list[Document]:
    """Load multilingual FAQ entries from faq.json."""
    faq_path = Path(faq_path)
    if not faq_path.is_file():
        print(f"FAQ file not found; skipping: {faq_path}")
        return []

    with faq_path.open("r", encoding="utf-8") as fp:
        faqs = json.load(fp)

    if not isinstance(faqs, list):
        raise ValueError("faq.json must contain a top-level JSON array.")

    language_fields = [
        ("en", "question_en", "answer_en", "en"),
        ("hi", "question_hi", "answer_hi", "hi"),
        ("hinglish", "question_hinglish", "answer_hinglish", "hinglish"),
    ]

    documents: list[Document] = []
    seen_ids: set[str] = set()

    for faq in faqs:
        faq_id = faq.get("id")
        if not faq_id:
            raise ValueError("Each FAQ must have a non-empty id.")
        if faq_id in seen_ids:
            raise ValueError(f"Duplicate FAQ id: {faq_id}")
        seen_ids.add(faq_id)

        for language, question_key, answer_key, answer_language in language_fields:
            question = faq.get(question_key)
            answer = faq.get(answer_key)

            # Fall back to English answer for Hinglish if answer_hinglish is absent.
            if language == "hinglish" and not answer:
                answer = faq.get("answer_en")
                answer_language = "en"

            if not question or not answer:
                continue

            content = (
                f"Question ({language}): {question.strip()}\n"
                f"Answer ({answer_language}): {answer.strip()}"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "faq_id": str(faq_id),
                        "category": faq.get("category", "general"),
                        "source": faq.get("source", "faq.json"),
                        "source_type": "faq",
                        "language": language,
                        "answer_language": answer_language,
                    },
                )
            )

    print(f"Created {len(documents)} multilingual FAQ documents.")
    return documents


def prepare_documents(pdf_documents: list[Document], faq_documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", ".", " ", ""],
        length_function=len,
        add_start_index=True)

    # Split long policy pages, but keep each FAQ as an intact Q&A document.
    policy_chunks = splitter.split_documents(pdf_documents)

    for index, chunk in enumerate(policy_chunks):
        chunk.metadata["chunk_type"] = "policy_chunk"
        chunk.metadata["chunk_id"] = f"policy_{index}"

    for index, doc in enumerate(faq_documents):
        doc.metadata["chunk_type"] = "faq"
        doc.metadata["chunk_id"] = f"faq_{index}"

    chunks = policy_chunks + faq_documents

    if not chunks:
        raise ValueError("No documents were loaded. Check the PDF directory and FAQ file.")

    print(f"Prepared {len(chunks)} chunks.")
    return chunks


def build_qdrant_index(chunks: list[Document]) -> None:
    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True})

    print("Building Qdrant index...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=str(QDRANT_PATH),
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    print(f"Qdrant index saved to: {QDRANT_PATH}")


def save_bm25_documents(chunks: list[Document]) -> None:
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    with BM25_PATH.open("wb") as file:
        pickle.dump(chunks, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"BM25 documents saved to: {BM25_PATH}")


def build_index() -> None:
    print("=" * 55)
    print("Vaani Knowledge Base Indexer")
    print("=" * 55)

    pdf_documents = load_pdfs(DATA_DIR)
    faq_documents = load_faqs(FAQ_PATH)
    chunks = prepare_documents(pdf_documents, faq_documents)

    save_bm25_documents(chunks)
    build_qdrant_index(chunks)

    print("\nIndexing completed successfully.")
    print(f"Total indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    build_index()