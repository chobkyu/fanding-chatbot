# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from tools import search_tool
from langchain.agents.agent_toolkits import create_retriever_tool

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

##########################################################################

# 1. standalone question 생성용 프롬프트
condense_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""
다음 대화 이력과 질문을 바탕으로 단일 질문으로 바꾸세요:
대화 이력:
{chat_history}
질문:
{question}
"""
)
question_generator = LLMChain(
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    prompt=condense_prompt
)

# 2. 답변용 프롬프트
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a customer support chatbot.
Based on the following information (Context), answer the user's question (Question) accurately and politely in Korean.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 3. ConversationalRetrievalChain 생성
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    retriever=vectordb.as_retriever(),
    condense_question_llm=question_generator.llm,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=False
)

# 4. 대화 이력을 반영하는 함수
def get_answer_with_history(query: str, history: list) -> str:
    result = chat_chain.invoke({
        "question": query,
        "chat_history": history
    })
    return result["answer"]

##################


retriver_tool = create_retriever_tool(
    retriever = vectordb.as_retriever(),
    name="vector_search",
    description="매뉴얼 문서 기반 질문에 답변할 때 사용합니다."
)

tools = [retriver_tool, search_tool]

agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

def get_agent_answer(query: str) -> str:
    result = agent.run(query)
    print(result)
    return result


@app.post("/ask")
async def ask(question: Question):
    result = get_answer(question.query)
    return {"answer": result}
