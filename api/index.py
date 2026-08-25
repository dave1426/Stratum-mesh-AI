import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh API", version="1.6.0")

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
    return {"status": "StratumMesh core is online", "version": "1.6.0"}

@app.post("/api/index")
def route_prompt(request: PromptRequest):
    last_error = None
    for model_name in FALLBACK_CHAIN:
        try:
            # Set a hard timeout parameter so it doesn't hang forever
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=500,
                timeout=10 # 10 seconds max per model try
            )
            ai_reply = response.choices[0].message.content
            return {
                "success": True,
                "routed_model": response.model,
                "response": ai_reply
            }
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            continue
            
    # Return the exact backend error to the UI instead of crashing/hanging
    return {
        "success": False,
        "detail": f"All models failed. Last error: {last_error}"
    }
