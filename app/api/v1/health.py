from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    message: str


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse: API health status
    """
    return HealthResponse(status="healthy", message="API is running")
