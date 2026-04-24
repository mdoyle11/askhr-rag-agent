from pydantic import BaseModel, Field
from typing import Literal

class RetrievalGrade(BaseModel):
    grade: Literal['relevant', 'irrelevant'] = Field(
        description=(
            "Whether the retrieved documents are relevant to the user's question. "
            "Return 'relevant' if any document directly addresses the question. "
            "Return 'irrelevant' if none of the documents contain useful information."
            )
        )

class QueryVariants(BaseModel):
    variants: list[str] = Field(
        description='Alternative phrasing of the user question.'
    )

class ReRanker(BaseModel):
    scores: list[int] = Field(description='Relevance score 0-10 for each document, in the same order as the input.')
