from askhr.agent.state import AgentState
from langgraph.graph import StateGraph, START, END
from askhr.agent.nodes import rewrite, retrieve, rerank, grade, generate, fallback

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node('rewriter', rewrite)
    graph.add_node('retriever', retrieve)
    graph.add_node('reranker', rerank)
    graph.add_node('grader', grade)
    graph.add_node('generator', generate)
    graph.add_node('fallback', fallback)

    graph.add_edge(START, "rewriter")
    graph.add_edge('rewriter', 'retriever')
    graph.add_edge('retriever', 'reranker')
    graph.add_edge("reranker", "grader")
    graph.add_edge('generator', END)
    graph.add_edge('fallback', END)

    return graph.compile()
