"""Config routes — read and update configuration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from config import get_default_config

router = APIRouter(tags=["config"])

# Keep a mutable config instance for generation param updates
_config = get_default_config()


class GenerationUpdate(BaseModel):
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    return _config.to_dict()


@router.patch("/config/generation")
async def update_generation(body: GenerationUpdate) -> Dict[str, Any]:
    if body.temperature is not None:
        _config.generation.temperature = body.temperature
    if body.top_k is not None:
        _config.generation.top_k = body.top_k
    if body.top_p is not None:
        _config.generation.top_p = body.top_p

    return {
        "temperature": _config.generation.temperature,
        "top_k": _config.generation.top_k,
        "top_p": _config.generation.top_p,
    }
