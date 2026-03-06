"""
FastAPI application entrypoint for the AI Quilt Generator.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.quilt import router as quilt_router

app = FastAPI(
    title="AI Quilt Generator",
    description="LangGraph-orchestrated quilt pattern generator powered by Gemini 2.5 Flash",
    version="1.0.0",
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(quilt_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
