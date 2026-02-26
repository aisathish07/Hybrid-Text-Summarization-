"""
FastAPI server for the Hybrid Summarization Pipeline.
Run this in Google Colab (or any remote server) and expose via ngrok.

Usage (Colab):
    !pip install fastapi uvicorn pyngrok
    !python api_server.py

Usage (Local):
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from main import run_summarization_pipeline

app = FastAPI(title="Hybrid Summarizer API", version="1.0")


class SummarizeRequest(BaseModel):
    text: str
    top_n: int = 5
    clusters: int = 3
    max_length: int = 150
    language: str = "en"


class SummarizeResponse(BaseModel):
    best_model: Optional[str]
    best_summary: Optional[str]
    extractive_list: list
    extractive_text: str
    clustered_list: list
    clustered_text: str
    candidates: dict
    scores: dict


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Hybrid Summarizer API is running"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        results = run_summarization_pipeline(
            text=req.text,
            top_n=req.top_n,
            clusters=req.clusters,
            max_length=req.max_length,
            language=req.language
        )

        # Ensure best_summary is not None
        if not results.get("best_summary"):
            results["best_summary"] = results.get("extractive_text", "No summary generated.")
            results["best_model"] = "extractive_fallback"

        return SummarizeResponse(**results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Auto-detect Colab and use ngrok
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        from pyngrok import ngrok
        public_url = ngrok.connect(8000)
        print("=" * 60)
        print(f"  🌐 PUBLIC URL: {public_url}")
        print("  Copy this URL into your Streamlit sidebar!")
        print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
