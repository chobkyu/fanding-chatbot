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
from langchain.agents import create_openai_functions_agent, AgentExecutor
# 프롬프트 정의
from langchain.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()
embedding = OpenAIEmbeddings()
vectordb = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

class Question(BaseModel):
    query: str


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
    description="웹에서 검색할 수 없는 질문이거나, 팬딩 서비스의 매뉴얼 문서 안에 있는 내용이라면 이 도구를 사용하세요. 예: 기능 설명, 설정 방법, 사용법 등"
)

tools = [retriver_tool, search_tool]

## 해당 방식은 간단하지만 프롬프트를 커스터마이징해서 직접 넣을 수 없음
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 팬딩 고객센터 AI 챗봇입니다. 고객의 질문에 친절하고 정확하게 대답하세요."),
    ("user", "{input}"),
    ("ai", "{agent_scratchpad}")
])

agent_custom = create_openai_functions_agent(
    llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0.7),
    tools=tools,
    prompt=prompt
)

# Executor 생성
agent_executor = AgentExecutor(
    agent=agent_custom, 
    tools=tools, 
    verbose=True,
    return_intermediate_steps=True,
    # output_key="output"
)

def get_agent_answer(query: str) -> str:
    # result = agent.run(query)
    result = agent_executor.invoke({"input":query})
    print(result)
    return result['output']


@app.post("/ask")
async def ask(question: Question):
    result = get_answer(question.query)
    return {"answer": result}
