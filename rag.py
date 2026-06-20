from ollama import chat, embed
import chromadb

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "ssc_ncert"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_collection(COLLECTION_NAME)


def retrieve(query, k=5):
    response = embed(
        model=EMBED_MODEL,
        input=query
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results["documents"][0]

    return documents


def build_prompt(question, context_chunks):
    context = "\n\n".join(context_chunks)

    return f"""
You are an SSC CGL tutor.

Answer ONLY using the provided context.

If the answer is not found in the context, say:

"Information not found in knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""


def ask(question):
    docs = retrieve(question)

    prompt = build_prompt(question, docs)

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
