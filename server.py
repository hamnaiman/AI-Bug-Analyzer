from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from main import app as bug_graph, provider_settings

BASE = Path(__file__).resolve().parent
app = FastAPI(title='AI Bug Report Analyzer')


class BugRequest(BaseModel):
    description: str = Field(min_length=10, max_length=6000)


@app.get('/')
def home():
    return FileResponse(BASE / 'index.html')


@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get('/health')
def health():
    provider, model = provider_settings()
    return {'provider': provider, 'model': model}


@app.post('/analyze')
def analyze(body: BugRequest):
    description = body.description.strip()
    if len(description) < 10:
        raise HTTPException(400, 'Please provide a little more detail about the bug.')
    try:
        result = bug_graph.invoke({'description': description})
        provider, model = provider_settings()
        return {**result, 'provider': provider, 'model': model}
    except RuntimeError as error:
        raise HTTPException(502, str(error))
    except Exception:
        raise HTTPException(500, 'Could not analyze this report. Please try again.')
