from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from typing import TypedDict, List, Union
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent

# 🔧 도구 임포트
from tools import search_tool
from main import retriver_tool
tools = [retriver_tool, search_tool]

# 🤖 LLM 준비
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 팬딩 고객센터 AI 챗봇입니다. 고객의 질문에 친절하고 정확하게 대답하세요."),
    ("user", "{input}"),
    ("ai", "{agent_scratchpad}")
])
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 📦 상태 정의
class AgentState(TypedDict):
    input: str
    chat_history: List[Union[HumanMessage, AIMessage]]
    output: str

# 🧠 시스템 프롬프트
system_message = SystemMessage(content="당신은 팬딩 고객센터 AI 챗봇입니다. 고객의 질문에 친절하고 정확하게 대답하세요.")

# 🧠 LangGraph용 실행 함수
def run_with_memory(state: AgentState) -> AgentState:
    user_input = HumanMessage(content=state["input"])
    messages = [system_message] + state["chat_history"] + [user_input]

    response = llm.invoke(messages)

    new_history = state["chat_history"] + [user_input, response]

    return {
        "input": "",
        "chat_history": new_history,
        "output": response.content
    }

# 🧱 그래프 구성
graph = StateGraph(AgentState)
graph.add_node("chatbot", RunnableLambda(run_with_memory))
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

app = graph.compile()

# 🪜 전역 히스토리 유지
chat_history: List[Union[HumanMessage, AIMessage]] = []

# 🎯 호출 함수
def get_agent_answer_graph_history(query: str) -> str:
    global chat_history

    state = {
        "input": query,
        "chat_history": chat_history,
        "output": ""
    }

    result = app.invoke(state)
    chat_history = result["chat_history"]
    return result["output"]
