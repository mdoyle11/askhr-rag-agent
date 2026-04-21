from askhr.agent.state import AgentState
from askhr.agent.schemas import RetrievalGrade
from askhr.retrieval.search import search
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command
from askhr.config import get_settings
from typing import Literal

settings = get_settings()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=settings.google_api_key.get_secret_value()
    )

grader_llm = llm.with_structured_output(RetrievalGrade)

GRADER_PROMPT = """You are a grader evaluating whether retrieved documents are relevant to a user question.

Question: {question}

Retrieved documents:
{docs}

Reply with 'relevant' if the documents contain information that helps answer the question, or 'irrelevant' if they don't.
"""
# Separator
GENERATE_PROMPT = """You are an HR assistant answering employee questions based on the company handbook.

Use ONLY the information in the provided context to answer. If the context doesn't contain enough information to answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

async def retrieve(state: AgentState) -> dict:
    retrieved_docs = await search(state['message'], top_k=5)
    return {'retrieved_docs': retrieved_docs}


async def grade(state: AgentState) -> Command[Literal['generator', 'fallback']]:
    docs_content = "\n\n---\n\n".join(
        doc.get("content", "") for doc in state["retrieved_docs"]
    )
    prompt = GRADER_PROMPT.format(question=state['message'], docs=docs_content)
    graded = await grader_llm.ainvoke(prompt)
    goto = 'generator' if graded.grade == 'relevant' else 'fallback'
    return Command(update={'retrieval_grade': graded.grade}, goto=goto)


async def generate(state: AgentState) -> dict:
    context = '\n\n---\n\n'.join(f"{doc.get('section','Unknown')}\n{doc.get('content', '')}" for doc in state['retrieved_docs'])
    prompt = GENERATE_PROMPT.format(question=state['message'], context=context)
    response = await llm.ainvoke(prompt)
    return {'answer': response.content}


def fallback(state: AgentState) -> dict:
    return {'answer': "I couldn't find information about that in the employee handbook. "
                      "I can only answer questions covered by the handbook — "
                      "try rephrasing, or ask about policies like attendance, PTO, dress code, or conduct."}