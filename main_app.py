import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retriever_engine import get_advanced_retriever


# ✅ Set API Key (better: export in terminal)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCVetBoloz7Fc_18oZeCRpwSAIs5DUhPcI"


def run_rag_system():
    # 🔹 1. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    # 🔹 2. Retriever
    retriever = get_advanced_retriever()

    # 🔹 3. STRONG PROMPT (CRITICAL)
    system_prompt = (
        "You are an expert AI/ML teacher and research assistant.\n\n"

        "STRICT RULES:\n"
        "1. First give a clear, general definition.\n"
        "2. Then explain in 2-3 structured bullet points.\n"
        "3. Then relate the answer to the given context.\n"
        "4. Do NOT mix unrelated topics.\n"
        "5. If answer is not in context, say 'I don't know'.\n"
        "6. Always cite sources like [Source: filename].\n\n"

        "CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 🔹 4. Format retrieved documents
    def format_docs(docs):
        if not docs:
            return "No relevant context found."

        formatted = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            content = doc.page_content.strip().replace("\n", " ")

            formatted.append(
                f"[Source: {source}]\n{content}"
            )

        return "\n\n".join(formatted)

    # 🔹 5. LCEL RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": lambda x: x["input"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 🔹 6. CLI LOOP
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

            print("\n💡 Answer:\n")
            print(answer)

        except Exception as e:
            print("\n❌ Error occurred:")
            print(str(e))


if __name__ == "__main__":
    run_rag_system()