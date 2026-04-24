from typing import Annotated, TypedDict
from operator import add

class AgentState(TypedDict, total=False):
    message: str
    query_variants: list[str]
    retrieved_docs: list[dict]
    retrieval_grade: str
    answer: str