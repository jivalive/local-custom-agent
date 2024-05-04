from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import httpx
import asyncio
import ollama

app = FastAPI()

# This semaphore limits the number of concurrent tasks
concurrent_task_limit = 5
semaphore = asyncio.Semaphore(concurrent_task_limit)

async def fetch_data_from_external_api(data):
    async with semaphore:  # This will block if the limit of concurrent tasks is reached
        try:
            async with httpx.AsyncClient(timeout=1200.0) as client:
                # response = ollama.generate(model=data['model'], prompt=data['prompt'])
                response = await client.post(
                    "http://local-mistral-ready:11434/api/generate",
                    json=data
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}

@app.post("/request-external")
async def request_external(model: str, prompt: str, stream: bool = Query(default=False)):
    data = {"model": model, "prompt": prompt, "stream": stream}
    result = await fetch_data_from_external_api(data)
    return JSONResponse(content=result)