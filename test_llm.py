# import google.generativeai as genai

# genai.configure(api_key="AIzaSyCVetBoloz7Fc_18oZeCRpwSAIs5DUhPcI")

# for m in genai.list_models():
#     print(m.name)
# test the api
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key="AIzaSyCVetBoloz7Fc_18oZeCRpwSAIs5DUhPcI"
)

print(llm.invoke("Hello"))