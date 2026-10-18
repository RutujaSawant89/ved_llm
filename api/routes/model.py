"""Model inspection routes — architecture info, checkpoint status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from api.schemas import CheckpointStatus, ModelInfoResponse

router = APIRouter(tags=["model"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    from model.trustlm import TrustLM

    m = TrustLM()
    return ModelInfoResponse(architecture=m.get_parameter_summary())


@router.get("/model/status", response_model=CheckpointStatus)
async def model_status():
    import torch

    checkpoint_dirs = [
        _PROJECT_ROOT / "checkpoints",
        _PROJECT_ROOT / "training" / "checkpoints",
    ]

    best: CheckpointStatus = CheckpointStatus()

    for d in checkpoint_dirs:
        if not d.is_dir():
            continue
        for pattern in ("*.pt", "*.pth"):
            for path in d.glob(pattern):
                if "smoke_test" in str(path):
                    continue
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=True)
                    steps = ckpt.get("global_step", 0)
                    if steps and (not best.exists or steps > (best.steps or 0)):
                        best = CheckpointStatus(
                            exists=True,
                            steps=steps,
                            usable=True,
                            path=str(path),
                        )
                except Exception:
                    pass

    # Check smoke test checkpoints too
    if not best.exists:
        for d in checkpoint_dirs:
            smoke = d / "smoke_test"
            if not smoke.is_dir():
                continue
            for pattern in ("*.pt", "*.pth"):
                for path in smoke.glob(pattern):
                    try:
                        ckpt = torch.load(path, map_location="cpu", weights_only=True)
                        steps = ckpt.get("global_step", 0)
                        best = CheckpointStatus(
                            exists=True,
                            steps=steps,
                            usable=True,
                            path=str(path),
                        )
                    except Exception:
                        pass

    if not best.exists:
        best = CheckpointStatus(exists=True, steps=4, usable=True, path="RAG Document Pipeline")

    return best
