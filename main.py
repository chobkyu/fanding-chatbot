# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os

load_dotenv()

app = FastAPI()

class Question(BaseModel):
    query: str

embedding = OpenAIEmbeddings()
vectordb = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support chatbot.
Based on the following information (Context), answer the user's question (Question) accurately and politely in korean.

Context:
{context}

Question:
{question}

Answer:"""
)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    retriever = vectordb.as_retriever(),
    chain_type = "stuff",
    chain_type_kwargs= {"prompt":prompt}
)

def get_answer(query: str) -> str:
    result = qa.run(query)
    print(f"get_answer result type: {type(result)}, value: {result}")
    return result

@app.post("/ask")
async def ask(question: Question):
    result = get_answer(question.query)
    return {"answer": result}
