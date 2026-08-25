import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh AI Router", version="1.1.0")

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "auto"

def select_target_model(prompt: str, task_type: str) -> str:
    """Dynamically select the model based on task intent or keywords."""
    text = prompt.lower()
    
    if task_type == "code" or any(kw in text for kw in ["code", "python", "html", "script", "function", "bug"]):
        # Route coding tasks to a strong coding/logic model
        return "deepseek/deepseek-chat" # or a specific free alternative available on OpenRouter
    elif task_type == "creative" or any(kw in text for kw in ["story", "write", "poem", "blog"]):
        return "openrouter/auto"
    else:
        # Default fallback route
        return "openrouter/auto"

@app.get("/")
def read_root():
    return {"status": "StratumMesh core is online with Intent Routing", "version": "1.1.0"}

@app.post("/api/route")
def route_prompt(request: PromptRequest):
    try:
        chosen_model = select_target_model(request.prompt, request.task_type)
        
        response = completion(
            model=chosen_model,
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
