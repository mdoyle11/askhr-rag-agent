from askhr.agent.base_llm import llm, RETRY_KWARGS
from askhr.agent.schemas import ReRanker

rerank_llm = llm.with_structured_output(ReRanker).with_retry(**RETRY_KWARGS)

RERANK_PROMPT = (
    "Score each document's relevance to the user's question on a scale of 0-10. "
    "Return a list of scores in the same order as the documents.\n\n"
    "Question: {query}\n\n"
    "Documents:\n{docs_block}"
)

async def rerank_docs(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:

    if not docs:
        return []
    docs_block = "\n\n".join(f"[{i + 1}] {d['content']}" for i, d in enumerate(docs))

    prompt = RERANK_PROMPT.format(query=query, docs_block=docs_block)
    response = await rerank_llm.ainvoke(prompt)

    scores = list(response.scores)
    if len(scores) < len(docs):
        scores.extend([0] * (len(docs) - len(scores)))

    reranked = [{**doc, 'rerank_score': score} for doc, score in zip(docs, scores)]
    reranked.sort(key=lambda d: d['rerank_score'], reverse=True)
    return reranked[:top_k]