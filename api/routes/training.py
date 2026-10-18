"""Training routes — start/stop training, status."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request

from api.schemas import TrainRequest, TrainStatusResponse

router = APIRouter(tags=["training"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.post("/train/start")
async def start_training(body: TrainRequest, request: Request):
    if request.app.state.train_process is not None:
        poll = request.app.state.train_process.poll()
        if poll is None:
            return {"status": "already_running"}

    cmd = [sys.executable, str(_PROJECT_ROOT / "training" / "train.py")]

    if body.mode == "smoke_test":
        cmd.append("--smoke-test")
    else:
        cmd.extend([
            "--epochs", str(body.epochs),
            "--batch-size", str(body.batch_size),
            "--lr", str(body.learning_rate),
        ])

    process = subprocess.Popen(
        cmd,
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    request.app.state.train_process = process
    return {"status": "started", "pid": process.pid, "mode": body.mode}


@router.get("/train/status", response_model=TrainStatusResponse)
async def train_status(request: Request):
    proc = request.app.state.train_process
    if proc is None:
        return TrainStatusResponse(running=False)

    poll = proc.poll()
    if poll is not None:
        # Process finished
        return TrainStatusResponse(running=False, step=0)

    return TrainStatusResponse(running=True)


@router.post("/train/stop")
async def stop_training(request: Request):
    proc = request.app.state.train_process
    if proc is None:
        return {"status": "not_running"}

    proc.terminate()
    request.app.state.train_process = None
    return {"status": "stopped"}
