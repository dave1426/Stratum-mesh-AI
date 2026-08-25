import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh API", version="1.7.0")

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "auto"

@app.get("/")
def read_root():
    return {"status": "StratumMesh core is online", "version": "1.7.0"}

@app.post("/api/index")
def route_prompt(request: PromptRequest):
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        return {
            "success": False,
            "detail": "API Key not found in Vercel environment variables."
        }

    # Direct API request to OpenRouter to avoid any SDK timeout overhead
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": 500
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15)
        data = response.json()
        
        if response.status_code == 200 and "choices" in data:
            ai_reply = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "routed_model": "deepseek/deepseek-chat (Direct)",
                "response": ai_reply
            }
        else:
            return {
                "success": False,
                "detail": f"API Error: {data.get('error', {}).get('message', 'Unknown error')}"
            }
    except Exception as e:
        return {
            "success": False,
            "detail": f"Request Exception: {str(e)}"
        }
