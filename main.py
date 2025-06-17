# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class Question(BaseModel):
    query: str

embedding = OpenAIEmbeddings()
vectordb = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    retriever = vectordb.as_retriever()
)

def get_answer(query: str) -> str:
    result = qa.run(query)
    print(f"get_answer result type: {type(result)}, value: {result}")
    return result

@app.post("/ask")
async def ask(question: Question):
    result = get_answer(question.query)
    return {"answer": result}
