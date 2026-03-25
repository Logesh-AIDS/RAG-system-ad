import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from process_papers import load_and_chunk_papers  # Import your previous logic

def create_vector_db():
    # 1. Get the chunks from Step 2
    chunks = load_and_chunk_papers()
    if not chunks:
        print("❌ No chunks to embed!")
        return

    # 2. Choose an Embedding Model (Completely FREE & Local)
    # 'all-MiniLM-L6-v2' is small, fast, and great for research papers
    print("🤖 Loading embedding model (Hugging Face)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Create the Vector Database (ChromaDB)
    # persist_directory: where it saves the database on your disk
    persist_directory = "chroma_db"
    
    print(f"📦 Creating Vector Database in '{persist_directory}'...")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # 4. Save (Persist) the database
    # In newer versions of Chroma, this happens automatically, but we ensure it here
    print(f"✅ Success! Vector DB created with {len(chunks)} chunks.")
    return vector_db

if __name__ == "__main__":
    create_vector_db()
