"""Health check route."""

from fastapi import APIRouter, Request

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(status="ok", model="TrustLM", device=device)
