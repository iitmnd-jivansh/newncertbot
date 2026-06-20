import os
from pathlib import Path

import chromadb
from pypdf import PdfReader
from ollama import embed
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================
# CONFIG
# ==========================

PDF_ROOT = "./NCERT"          # folder containing PDFs
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ssc_ncert"

EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ==========================
# CHROMA
# ==========================

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# ==========================
# TEXT SPLITTER
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# ==========================
# PDF EXTRACTION
# ==========================

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                pages.append(text)
        except Exception as e:
            print(f"Error reading page in {pdf_path}: {e}")

    return "\n".join(pages)

# ==========================
# INGEST ONE PDF
# ==========================

def ingest_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path}")

    text = extract_pdf_text(pdf_path)

    if not text.strip():
        print("No text found.")
        return

    chunks = splitter.split_text(text)

    print(f"Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        try:
            response = embed(
                model=EMBED_MODEL,
                input=chunk
            )

            vector = response["embeddings"][0]

            collection.add(
                ids=[f"{pdf_path}_{i}"],
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{
                    "source": str(pdf_path)
                }]
            )

        except Exception as e:
            print(f"Chunk {i} failed: {e}")

# ==========================
# MAIN
# ==========================

def main():
    pdfs = list(Path(PDF_ROOT).rglob("*.pdf"))

    print(f"Found {len(pdfs)} PDFs")

    for pdf in pdfs:
        ingest_pdf(pdf)

    print("\nIngestion complete.")

if __name__ == "__main__":
    main()
