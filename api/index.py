import os
import traceback
from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="StratumMesh API", version="1.8.0")

class PromptRequest(BaseModel):
    prompt: str
    task_type: str = "auto"

@app.get("/")
def read_root():
    return {"status": "StratumMesh core is online", "version": "1.8.0"}

@app.post("/api/index")
def route_prompt(request: PromptRequest):
    try:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        
        if not api_key:
            return {
                "success": False,
                "detail": "Error: API Key is missing from Vercel environment variables."
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": 500
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
        
        # If OpenRouter returns non-200, return the text error
        if response.status_code != 200:
            return {
                "success": False,
                "detail": f"OpenRouter HTTP {response.status_code}: {response.text}"
            }
            
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            ai_reply = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "routed_model": "deepseek/deepseek-chat",
                "response": ai_reply
            }
        else:
            return {
                "success": False,
                "detail": f"Unexpected API response structure: {data}"
            }

    except Exception as e:
        # This will catch the exact error and print it to your chat UI!
        return {
            "success": False,
            "detail": f"Server Exception: {str(e)}"
        }
