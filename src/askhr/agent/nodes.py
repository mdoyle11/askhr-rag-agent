import asyncio
from askhr.agent.base_llm import llm, RETRY_KWARGS
from askhr.agent.state import AgentState
from askhr.agent.schemas import RetrievalGrade
from askhr.retrieval.search import search
from askhr.retrieval.rewrite import generate_query_variants, reciprocal_rank_fusion, generate_hypothetical_answer
from askhr.retrieval.rerank import rerank_docs
from langgraph.types import Command
from askhr.config import get_settings
from typing import Literal

settings = get_settings()

grader_llm = llm.with_structured_output(RetrievalGrade).with_retry(**RETRY_KWARGS)
generater_llm = llm.with_retry(**RETRY_KWARGS)

GRADER_PROMPT = ("You are a grader evaluating whether retrieved documents are relevant to a user question.\n"
"Question: {question}\n\n"
"Retrieved documents:\n"
"{docs}\n\n"
"Reply with 'relevant' if the documents contain information that helps answer the question, or 'irrelevant' if they don't.")

GENERATE_PROMPT = ("You are an HR assistant answering employee questions based on the company handbook.\n"
"Use ONLY the information in the provided context to answer. If the context doesn't contain enough information to answer, say so clearly.\n"
"Context:\n"
"{context}\n\n"
"Question: {question}\n\n"
"Answer:")

async def rewrite(state: AgentState) -> list[str]:
    if settings.rewrite_method == 'multi-query':
        variants = await generate_query_variants(state['message'], n=3)
        return {'query_variants': variants}
    hyde = await generate_hypothetical_answer(state['message'])
    return {'query_variants': hyde}

async def retrieve(state: AgentState) -> dict:
    search_queries = [search(variant, top_k=10) for variant in state['query_variants']]
    ranked_lists = await asyncio.gather(*search_queries)
    fused = reciprocal_rank_fusion(ranked_lists, top_k=20)
    return {'retrieved_docs': fused}

async def rerank(state: AgentState) -> dict:
    reranked = await rerank_docs(state['message'], state['retrieved_docs'], top_k=5)
    return {'retrieved_docs': reranked}

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