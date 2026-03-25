# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from retriever_engine import get_advanced_retriever

# # 1. Setup your API Key
# os.environ["Gemini API Key"] = "AIzaSyCVetBoloz7Fc_18oZeCRpwSAIs5DUhPcI"

# def run_rag_system():
#     # 2. Initialize the LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

#     # 3. Get the Retriever from Step 4
#     retriever = get_advanced_retriever()

#     # 4. Define the Prompt
#     system_prompt = (
#         "You are an expert research assistant. Use the provided context to answer the user's question. "
#         "If you don't know the answer based on the context, say that you don't know. "
#         "Keep the answer concise and professional. Always cite the 'Source' filename."
#         "\n\n"
#         "{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])

#     # 5. Create the RAG Chain
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})

#     # 6. Interaction Loop
#     print("\n🚀 RAG System Ready! Ask your research questions (type 'exit' to quit).")
#     while True:
#         query = input("\n👤 Question: ")
#         if query.lower() == 'exit': break
        
#         print("🤖 Thinking...")
#         response = rag_chain.invoke({"input": query})
        
#         print(f"\n💡 Answer: {response['answer']}")
        
#         # Show which papers were used for transparency
#         sources = set([doc.metadata.get('source_file') for doc in response['context']])
#         print(f"📚 Sources used: {', '.join(sources)}")

# if __name__ == "__main__":
#     run_rag_system()


# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

# from retriever_engine import get_advanced_retriever


# # ✅ 1. Set API Key (IMPORTANT)
# os.environ["GOOGLE_API_KEY"] = "AIzaSyCVetBoloz7Fc_18oZeCRpwSAIs5DUhPcI"


# def run_rag_system():
#     # ✅ 2. Initialize LLM
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0
#     )

#     # ✅ 3. Retriever
#     retriever = get_advanced_retriever()

#     # ✅ 4. Prompt Template (0.2.x style)
#     prompt_template = """
# You are an expert research assistant.

# Use ONLY the provided context to answer the question.
# If the answer is not in the context, say "I don't know".

# Keep answers concise and professional.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"]
#     )

#     # ✅ 5. RAG Chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt}
#     )

#     # ✅ 6. Interaction Loop
#     print("\n🚀 RAG System Ready! (type 'exit' to quit)")

#     while True:
#         query = input("\n👤 Question: ")
#         if query.lower() == "exit":
#             break

#         print("🤖 Thinking...")

#         result = qa_chain.invoke({"query": query})

#         print(f"\n💡 Answer:\n{result['result']}")

#         # ✅ Sources
#         sources = set(
#             doc.metadata.get("source_file", "Unknown")
#             for doc in result["source_documents"]
#         )

#         print(f"\n📚 Sources used: {', '.join(sources)}")


# if __name__ == "__main__":
#     run_rag_system()



import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever_engine import get_advanced_retriever


# ✅ BEST PRACTICE: set in terminal instead of hardcoding
# export GOOGLE_API_KEY="your_key"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCVetBoloz7Fc_18oZeCRpwSAIs5DUhPcI"


def run_rag_system():
    # 🔹 1. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 🔹 2. Retriever (must return a runnable retriever)
    retriever = get_advanced_retriever()

    # 🔹 3. Prompt
    system_prompt = (
        "You are an expert research assistant.\n"
        "Use ONLY the provided context to answer.\n"
        "If the answer is not in context, say 'I don't know'.\n"
        "Keep answers concise and factual.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 🔹 4. Format documents safely
    def format_docs(docs):
        if not docs:
            return "No relevant context found."

        return "\n\n".join(
            f"Source: {doc.metadata.get('source_file', 'Unknown')}\n{doc.page_content}"
            for doc in docs
        )

    # 🔹 5. RAG Chain (LCEL)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 🔹 6. CLI loop
    print("\n🚀 RAG System Ready! Type 'exit' to quit.")

    while True:
        query = input("\n👤 Question: ").strip()

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        if not query:
            print("⚠️ Please enter a valid question.")
            continue

        try:
            print("🤖 Thinking...")
            answer = rag_chain.invoke({"input": query})

            print("\n💡 Answer:")
            print(answer)

        except Exception as e:
            print("\n❌ Error occurred:")
            print(str(e))


if __name__ == "__main__":
    run_rag_system()