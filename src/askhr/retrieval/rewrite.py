from askhr.agent.base_llm import llm
from askhr.agent.schemas import QueryVariants
from collections import defaultdict

multi_query_llm = llm.with_structured_output(QueryVariants)

MULTI_QUERY_PROMPT = ("You rephrase user questions to improve document retrieval. "
"Given the question, generate {n} alternative phrasings that express the same intent "
"using different vocabulary and sentence structures. "
"Keep each variant concise and specific. Do not answer the question.\n\n"
"Question: {question}")

HYPOTHETICAL_PROMPT = """Write a short, plausible answer to the following question as if it appeared in
an employee handbook. 2-4 sentences. Do not hedge or disclaim — write it as
authoritative policy text. This text will be used to find the real policy,
not shown to anyone.

Question: {question}"""


async def generate_query_variants(question: str, n: int = 3) -> list[str]:
    prompt = MULTI_QUERY_PROMPT.format(question=question, n=n)
    results = await multi_query_llm.ainvoke(prompt)
    return [question, *results.variants]

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
    ) -> list[dict]:

    scores: dict[tuple, float] = defaultdict(float)
    docs_by_key: dict[tuple, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = (doc['source'], doc['chunk_index'])
            scores[key] += 1 / (k + rank)
            docs_by_key.setdefault(key, doc)
    top_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**docs_by_key[key], 'score': scores[key]} for key in top_keys]

async def generate_hypothetical_answer(question: str) -> str:
    prompt = HYPOTHETICAL_PROMPT.format(question=question)
    response = await llm.ainvoke(prompt)
    return response.content