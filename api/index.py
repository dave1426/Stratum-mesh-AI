import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh AI Router", version="1.2.0")

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "auto"

# Define a prioritized chain of reliable free/general models on OpenRouter
FALLBACK_CHAIN = [
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v4-flash-0731",
    "openrouter/auto"
]

@app.get("/")
def read_root():
    return {"status": "StratumMesh core is online with Fallback Resilience", "version": "1.2.0"}

@app.post("/api/route")
def route_prompt(request: PromptRequest):
    last_error = None
    
    # Iterate through the fallback chain until one succeeds
    for model_name in FALLBACK_CHAIN:
        try:
            response = completion(
                model=model_name,
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
            last_error = str(e)
            continue # Try the next model in the chain
            
    # If all models in the chain fail
    raise HTTPException(status_code=500, detail=f"All fallback models failed. Last error: {last_error}")
