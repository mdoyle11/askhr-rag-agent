from askhr.agent.state import AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from askhr.config import get_settings

settings = get_settings()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=settings.google_api_key.get_secret_value()
    )

async def respond(state: AgentState) -> dict:
    response = await llm.ainvoke(state['message'])
    return {'answer': response.content}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("response", respond)

    graph.add_edge(START, "response")
    graph.add_edge("response", END)

    return graph.compile()
