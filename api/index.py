import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh Streaming Router", version="1.3.0")

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "auto"

FALLBACK_CHAIN = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v4-flash-0731",
    "openrouter/auto"
]

@app.get("/")
def read_root():
    return {"status": "StratumMesh core is online with Streaming", "version": "1.3.0"}

@app.post("/api/route")
def route_prompt(request: PromptRequest):
    def generate():
        success = False
        for model_name in FALLBACK_CHAIN:
            try:
                # Request streaming from litellm
                response = completion(
                    model=model_name,
                    messages=[{"role": "user", "content": request.prompt}],
                    max_tokens=1000,
                    stream=True
                )
                
                # Send model info first
                yield json.dumps({"type": "meta", "routed_model": model_name}) + "\n"
                
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield json.dumps({"type": "content", "text": delta}) + "\n"
                success = True
                break
            except Exception as e:
                continue
                
        if not success:
            yield json.dumps({"type": "error", "detail": "All fallback models failed during stream."}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
