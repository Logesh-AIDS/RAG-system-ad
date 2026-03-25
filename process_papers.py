import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_and_chunk_papers(directory="data"):
    # 1. Initialize the Smart Splitter
    # chunk_size: 1000 characters is a good balance for research papers
    # chunk_overlap: 200 chars ensures context flows between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )

    all_chunks = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]

    if not pdf_files:
        print("❌ No PDFs found in the data folder!")
        return []

    print(f"📄 Found {len(pdf_files)} papers. Starting processing...")

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        
        try:
            # Load PDF (PyMuPDF is fast and handles math better)
            loader = PyMuPDFLoader(file_path)
            data = loader.load()

            # Split into chunks
            chunks = text_splitter.split_documents(data)
            
            # Add metadata so we know WHICH paper this chunk came from
            for chunk in chunks:
                chunk.metadata["source_file"] = filename
            
            all_chunks.extend(chunks)
            print(f"✅ Processed '{filename}': Created {len(chunks)} chunks.")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n🎯 Total Chunks Created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk_papers()
    # Let's peek at the first chunk to see if it looks right
    if chunks:
        print("\n--- SAMPLE CHUNK ---")
        print(chunks[0].page_content[:200] + "...")
        print(f"Source: {chunks[0].metadata['source_file']}")
# if __name__ == "__main__":
#     # This only runs if you run process_papers.py directly
#     load_and_chunk_papers()

