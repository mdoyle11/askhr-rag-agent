from typing import Annotated, TypedDict
from operator import add

class AgentState(TypedDict, total=False):
    message: str
    retrieved_docs: Annotated[list[dict], add]
    retrieval_grade: str
    answer: str

