from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.chat_models import ChatOpenAI
from typing import TypedDict, List, Union
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.agent import AgentFinish  # 🔧 추가

# 🔧 도구 임포트
from tools import search_tool
from main import retriver_tool
tools = [retriver_tool, search_tool]

# 🤖 LLM 및 Agent 세팅
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
parser = OpenAIFunctionsAgentOutputParser()

# 📦 상태 정의
class AgentState(TypedDict):
    input: str
    chat_history: List[Union[HumanMessage, AIMessage, ToolMessage]]
    output: str

# 🧠 시스템 프롬프트
system_message = SystemMessage(content="당신은 팬딩 고객센터 AI 챗봇입니다. 고객의 질문에 친절하고 정확하게 대답하세요.")

# 🔧 intermediate_steps 추출 함수
def extract_intermediate_steps(messages: List[Union[HumanMessage, AIMessage, ToolMessage]]):
    steps = []
    for i in range(len(messages) - 1):
        if isinstance(messages[i], AIMessage) and isinstance(messages[i + 1], ToolMessage):
            steps.append((messages[i], messages[i + 1]))
    return steps

# 🧠 LangGraph 실행 함수
def run_with_memory_and_tools(state: AgentState) -> AgentState:
    user_input = HumanMessage(content=state["input"])
    history = state["chat_history"]
    intermediate_steps = extract_intermediate_steps(history)

    agent_output = agent.invoke({
        "input": state["input"],
        "intermediate_steps": intermediate_steps
    })

    print(agent_output)
    new_history = history + [user_input]

    # 🔧 AgentFinish 응답 처리
    if isinstance(agent_output, AgentFinish):
        final_msg = AIMessage(content=agent_output.return_values['output'])
        new_history.append(final_msg)
        return {
            "input": "",
            "chat_history": new_history,
            "output": final_msg.content
        }

    # 🔧 tool calling 처리
    if isinstance(agent_output, AIMessage) and "tool_calls" in agent_output.additional_kwargs:
        tool_messages = []
        for tool_call in agent_output.additional_kwargs["tool_calls"]:
            func_name = tool_call["function"]["name"]
            arguments = eval(tool_call["function"]["arguments"])
            for t in tools:
                if t.name == func_name:
                    result = t.func(**arguments)
                    tool_messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=str(result)
                    ))

        # 🔧 최종 LLM 응답
        final_messages = [system_message] + new_history + [agent_output] + tool_messages
        final_response = llm.invoke(final_messages)

        new_history += [agent_output] + tool_messages + [final_response]

        return {
            "input": "",
            "chat_history": new_history,
            "output": final_response.content
        }

    # 🔧 fallback: 그냥 AIMessage
    if isinstance(agent_output, AIMessage):
        new_history.append(agent_output)
        return {
            "input": "",
            "chat_history": new_history,
            "output": agent_output.content
        }

    # 🔧 예외 처리
    return {
        "input": "",
        "chat_history": new_history,
        "output": "예상치 못한 오류가 발생했습니다."
    }

# 🧱 그래프 구성
graph = StateGraph(AgentState)
graph.add_node("chatbot", RunnableLambda(run_with_memory_and_tools))
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")
app = graph.compile()

# 🪜 전역 히스토리
chat_history: List[Union[HumanMessage, AIMessage, ToolMessage]] = []

# 🎯 호출 함수
def get_agent_answer_graph_history(query: str) -> str:
    global chat_history

    state = {
        "input": query,
        "chat_history": chat_history,
        "output": ""
    }

    result = app.invoke(state)
    print(result)
    chat_history = result["chat_history"]
    return result["output"]
