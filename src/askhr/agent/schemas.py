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