import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh AI Router", version="1.0.0")

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "general"

@app.get("/")
def read_root():
    return {"status": "StratumMesh core is online", "version": "1.0.0"}

@app.post("/api/route")
def route_prompt(request: PromptRequest):
    try:
        response = completion(
            model="openrouter/auto",
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=1000
        )

        ai_reply = response.choices[0].message.content
        return {
            "success": True,
            "routed_model": response.model,
            "response": ai_reply
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
