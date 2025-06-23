# langgraph 기반 상태 기억 Agent
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# 전역 상태 저장용 (간단 구현)
chat_history = []

# 도구 리스트 정의 (기존 tools 그대로 사용)
from tools import search_tool
from main import retriver_tool
tools = [retriver_tool, search_tool]

# 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 팬딩 고객센터 AI 챗봇입니다. 고객의 질문에 친절하고 정확하게 대답하세요."),
    ("user", "{input}"),
    ("ai", "{agent_scratchpad}")
])

# LLM + Agent 생성
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
agent_custom = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor
executor = AgentExecutor(
    agent=agent_custom, 
    tools=tools, 
    verbose=True,
)

# LangGraph용 상태 정의
from typing import TypedDict, List, Union

class AgentState(TypedDict):
    input: str
    chat_history: List[Union[HumanMessage, AIMessage]]

# LangGraph에서 실행할 함수
def run_agent_with_history(state: AgentState) -> AgentState:
    result = executor.invoke({
        "input": state["input"],
        "chat_history": state["chat_history"]
    })

    # 히스토리 업데이트
    new_history = state["chat_history"] + [
        HumanMessage(content=state["input"]),
        AIMessage(content=result["output"])
    ]

    return {
        "input": "",
        "chat_history": new_history,
        "output": result["output"]
    }

# LangGraph 구성
graph = StateGraph(AgentState)
graph.add_node("agent", RunnableLambda(run_agent_with_history))
graph.set_entry_point("agent")
graph.set_finish_point("agent")

# 컴파일
app = graph.compile()

# 👉 함수처럼 쓰는 래퍼
def get_agent_answer_graph(query: str) -> str:
    global chat_history

    state = {
        "input": query,
        "chat_history": chat_history
    }

    result = app.invoke(state)
    print(result)
    chat_history = result["chat_history"]
    return result["chat_history"][-1].content
