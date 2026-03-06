"""
FastAPI route for quilt generation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.langgraph_flow import run_quilt_flow

router = APIRouter()


class QuiltRequest(BaseModel):
    """Input contract for quilt generation."""
    width: float = Field(..., gt=0, description="Canvas width (must be > 0)")
    height: float = Field(..., gt=0, description="Canvas height (must be > 0)")
    style: str = Field(
        "classic_patchwork",
        description="Pattern style: classic_patchwork, star_burst, log_cabin, flying_geese, crazy_quilt",
    )


@router.post("/generate-quilt")
async def generate_quilt(request: QuiltRequest):
    """Generate a quilt pattern using LangGraph + Gemini.

    Accepts width and height, runs the full AI workflow,
    and returns ONLY valid quilt JSON. Never returns invalid output.
    """
    try:
        quilt = run_quilt_flow(
            width=int(request.width),
            height=int(request.height),
            style=request.style,
        )
        return quilt
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quilt generation failed: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {e}",
        )
