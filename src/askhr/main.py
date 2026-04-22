from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from askhr.agent.graph import build_graph
from pydantic import BaseModel

app = FastAPI()
graph = build_graph()

try:
    from askhr.sandbox_mount import mount as _mount_sandbox(app)
except ImportError:
    pass

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def send_response(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in graph.astream_events({'message': request.message}):
                if event['event'] == 'on_chat_model_stream' and event['metadata'].get('langgraph_node') == 'generator':
                    chunk = event['data']['chunk']
                    token = chunk.content
                    if token:
                        payload = json.dumps({'type': 'token', 'content': token})
                        yield f'data: {payload}\n\n'
                elif event['event'] == 'on_chain_end' and event['metadata'].get('langgraph_node') == 'fallback':
                    answer = event['data']['output'].get('answer')
                    if answer:
                        payload = json.dumps({'type': 'token', 'content': answer})                  
                        yield f'data: {payload}\n\n'
                elif event['event'] == 'on_chain_end' and event['metadata'].get('langgraph_node') == 'retriever':
                    output = event['data'].get('output', {})
                    docs = output.get('retrieved_docs', [])
                    sections = [section.get('section') for section in docs]
                    payload = json.dumps({'type': 'sources', 'content': sections})
                    yield f'data: {payload}\n\n'
                    
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import logging
            logging.exception('Streaming error')
            payload = json.dumps({'type': 'error', 'content': 'An error occurred while processing your request'})
            yield f'data: {payload}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")
