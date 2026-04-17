from fastapi import FastAPI
from askhr.agent.graph import build_graph
from pydantic import BaseModel

app = FastAPI()
graph = build_graph()

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def send_response(request: ChatRequest) -> ChatResponse:
    
    response = await graph.ainvoke({'message': request.message})
    return ChatResponse(answer=response['answer'])
