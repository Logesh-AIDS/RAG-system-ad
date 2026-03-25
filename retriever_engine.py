# import os
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# # CORRECT UPDATED IMPORTS
# from langchain.retrievers import EnsembleRetriever
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain_community.document_compressors.flashrank import FlashrankRerank
# from langchain_community.retrievers import BM25Retriever


# from process_papers import load_and_chunk_papers

# import os
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# # DIRECT IMPORTS (More stable in newer LangChain versions)
# # --- CORRECTED IMPORTS ---
# # Core retrievers live in the main 'langchain' package
# from langchain_community.retrievers import EnsembleRetriever
# from langchain_community.retrievers import ContextualCompressionRetriever

# # Specific community tools live in 'langchain_community'
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.document_compressors import EmbeddingCompressor


# from process_papers import load_and_chunk_papers

# import os

# from langchain_community.vectorstores import Chroma
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Core retrievers
# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# from sentence_transformers import CrossEncoder

# from process_papers import load_and_chunk_papers


# # -------------------------------
# # 🔁 LOAD DATA ONCE (Performance Fix)
# # -------------------------------
# print("📄 Loading and chunking papers...")
# chunks = load_and_chunk_papers()


# # -------------------------------
# # 🚀 MAIN RETRIEVER FUNCTION
# # -------------------------------
# def get_advanced_retriever():
#     # 1. Embeddings
#     embeddings = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )

#     # 2. Vector DB (Chroma)
#     vectorstore = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     vector_retriever = vectorstore.as_retriever(
#         search_kwargs={"k": 10}
#     )

#     # 3. BM25 (Keyword Search)
#     keyword_retriever = BM25Retriever.from_documents(chunks)
#     keyword_retriever.k = 10

#     # 4. Hybrid Retriever (Ensemble)
#     ensemble_retriever = EnsembleRetriever(
#         retrievers=[vector_retriever, keyword_retriever],
#         weights=[0.5, 0.5]
#     )

#     # 5. 🔥 RERANKER (CRITICAL)
#     model = HuggingFaceCrossEncoder(
#     model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
#     )

#     compressor = CrossEncoderReranker(
#         model=model,
#         top_n=5
#     )

#     compression_retriever = ContextualCompressionRetriever(
#         base_retriever=ensemble_retriever,
#         base_compressor=compressor
#     )

#     return compression_retriever


# # -------------------------------
# # 🧪 TESTING ENTRY POINT
# # -------------------------------
# if __name__ == "__main__":
#     retriever = get_advanced_retriever()

#     while True:
#         query = input("\n🔍 Ask a technical question (or type 'exit'): ")
#         if query.lower() == "exit":
#             break

#         results = retriever.invoke(query)

#         print(f"\n🎯 Top {len(results)} Relevant Chunks:\n")

#         for i, doc in enumerate(results, 1):
#             print(f"--- Result {i} ---")
#             print(f"📁 Source: {doc.metadata.get('source_file', 'Unknown')}")
#             print(f"🧠 Content:\n{doc.page_content[:300]}...\n")

# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# from langchain_core.runnables import Runnable

# from rank_bm25 import BM25Okapi

# from process_papers import load_and_chunk_papers


# # -------------------------------
# # 🔁 LOAD DATA ONCE
# # -------------------------------
# print("📄 Loading and chunking papers...")
# chunks = load_and_chunk_papers()


# # -------------------------------
# # 🔧 CUSTOM BM25 RETRIEVER
# # -------------------------------
# class BM25CustomRetriever(Runnable):
#     def __init__(self, documents, k=10):
#         self.docs = documents
#         self.k = k
#         self.tokenized_docs = [doc.page_content.split() for doc in documents]
#         self.bm25 = BM25Okapi(self.tokenized_docs)

#     def invoke(self, query, config=None):
#         tokenized_query = query.split()
#         scores = self.bm25.get_scores(tokenized_query)

#         # ✅ FIX: convert numpy → python list
#         pairs = list(zip(self.docs, scores.tolist()))

#         ranked = sorted(
#             pairs,
#             key=lambda x: float(x[1]),
#             reverse=True
#         )

#         return [doc for doc, _ in ranked[:self.k]]


# # -------------------------------
# # 🔧 HYBRID RETRIEVER
# # -------------------------------
# class HybridRetriever(Runnable):
#     def __init__(self, vector_retriever, keyword_retriever, k=10):
#         self.vector_retriever = vector_retriever
#         self.keyword_retriever = keyword_retriever
#         self.k = k

#     def invoke(self, query, config=None):
#         docs1 = self.vector_retriever.invoke(query)
#         docs2 = self.keyword_retriever.invoke(query)

#         seen = set()
#         combined = []

#         for doc in docs1 + docs2:
#             content = doc.page_content
#             if content not in seen:
#                 seen.add(content)
#                 combined.append(doc)

#         return combined[:self.k]


# # -------------------------------
# # 🚀 MAIN RETRIEVER
# # -------------------------------
# def get_advanced_retriever():
#     # 🔹 Embeddings
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     # 🔹 Vector DB
#     vectorstore = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     vector_retriever = vectorstore.as_retriever(
#         search_kwargs={"k": 10}
#     )

#     # 🔹 BM25
#     keyword_retriever = BM25CustomRetriever(chunks, k=10)

#     # 🔹 Hybrid
#     hybrid_retriever = HybridRetriever(
#         vector_retriever,
#         keyword_retriever,
#         k=10
#     )

#     # 🔹 CrossEncoder Reranker
#     cross_encoder = HuggingFaceCrossEncoder(
#         model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
#     )

#     compressor = CrossEncoderReranker(
#         model=cross_encoder,
#         top_n=5
#     )

#     # 🔹 Final Retriever
#     compression_retriever = ContextualCompressionRetriever(
#         base_retriever=hybrid_retriever,
#         base_compressor=compressor
#     )

#     return compression_retriever


# # -------------------------------
# # 🧪 TESTING
# # -------------------------------
# if __name__ == "__main__":
#     retriever = get_advanced_retriever()

#     while True:
#         query = input("\n🔍 Ask a technical question (or type 'exit'): ")

#         if query.lower() == "exit":
#             break

#         results = retriever.invoke(query)

#         print(f"\n🎯 Top {len(results)} Relevant Chunks:\n")

#         for i, doc in enumerate(results, 1):
#             print(f"--- Result {i} ---")
#             print(f"📁 Source: {doc.metadata.get('source_file', 'Unknown')}")
#             print(f"🧠 Content:\n{doc.page_content[:300]}...\n")
import os
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_core.runnables import Runnable

from rank_bm25 import BM25Okapi

from process_papers import load_and_chunk_papers


# -------------------------------
# 🔁 LOAD DATA ONCE
# -------------------------------
print("📄 Loading and chunking papers...")
chunks = load_and_chunk_papers()


# -------------------------------
# 🔧 CUSTOM BM25 RETRIEVER
# -------------------------------
class BM25CustomRetriever(Runnable):
    def __init__(self, documents, k=10):
        self.docs = documents
        self.k = k
        self.tokenized_docs = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def invoke(self, input, config=None):
        query = input if isinstance(input, str) else input["input"]

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        pairs = list(zip(self.docs, scores.tolist()))

        ranked = sorted(
            pairs,
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [doc for doc, _ in ranked[:self.k]]


# -------------------------------
# 🔧 HYBRID RETRIEVER
# -------------------------------
class HybridRetriever(Runnable):
    def __init__(self, vector_retriever, keyword_retriever, k=10):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.k = k

    def invoke(self, input, config=None):
        query = input if isinstance(input, str) else input["input"]

        docs1 = self.vector_retriever.invoke(query)
        docs2 = self.keyword_retriever.invoke(query)

        seen = set()
        combined = []

        for doc in docs1 + docs2:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                combined.append(doc)

        return combined[:self.k]


# -------------------------------
# 🔥 FIXED + SAFE RERANKER
# -------------------------------
class FixedCrossEncoderReranker(CrossEncoderReranker):
    model: Any

    def compress_documents(self, documents, query, callbacks=None):
        # Ensure query is string
        if not isinstance(query, str):
            query = str(query)

        clean_docs = []
        pairs = []

        for doc in documents:
            text = doc.page_content

            # Skip invalid content
            if not isinstance(text, str) or not text.strip():
                continue

            clean_docs.append(doc)
            pairs.append((query, text))

        if not pairs:
            return []

        # Correct API
        scores = self.model.score(pairs)

        doc_scores = list(zip(clean_docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_scores[:self.top_n]]


# -------------------------------
# 🚀 MAIN RETRIEVER
# -------------------------------
def get_advanced_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    keyword_retriever = BM25CustomRetriever(chunks, k=10)

    hybrid_retriever = HybridRetriever(
        vector_retriever,
        keyword_retriever,
        k=10
    )

    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    compressor = FixedCrossEncoderReranker(
        model=cross_encoder,
        top_n=5
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=compressor
    )

    return compression_retriever


# -------------------------------
# 🧪 TESTING
# -------------------------------
if __name__ == "__main__":
    retriever = get_advanced_retriever()

    while True:
        query = input("\n🔍 Ask a technical question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        try:
            results = retriever.invoke(query)

            print(f"\n🎯 Top {len(results)} Relevant Chunks:\n")

            for i, doc in enumerate(results, 1):
                print(f"--- Result {i} ---")
                print(f"📁 Source: {doc.metadata.get('source_file', 'Unknown')}")
                print(f"🧠 Content:\n{doc.page_content[:300]}...\n")

        except Exception as e:
            print("\n❌ Error occurred:")
            print(str(e))