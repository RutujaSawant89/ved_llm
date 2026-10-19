"""Training routes — start/stop training, live telemetry, and log streaming."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

from api.schemas import TrainRequest, TrainStatusResponse

router = APIRouter(tags=["training"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared telemetry state for the active training run
_telemetry_lock = threading.Lock()
_telemetry_state: Dict[str, Any] = {
    "running": False,
    "step": 0,
    "max_steps": 1000,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "perplexity": 0.0,
    "tokens_processed": 0,
    "logs": [],
}


def _stdout_reader_worker(proc: subprocess.Popen, max_steps: int) -> None:
    """Read process stdout line by line and parse training telemetry."""
    step_pattern = re.compile(
        r"\[Step\s+(\d+)\]\s+loss=([\d\.]+)\s+lr=([\d\.e\-]+)\s+tokens=([\d,]+)"
    )
    eval_pattern = re.compile(
        r"\[Eval Step\s+(\d+)\]\s+val_loss=([\d\.]+)\s+perplexity=([\d\.]+)"
    )

    with _telemetry_lock:
        _telemetry_state["running"] = True
        _telemetry_state["max_steps"] = max_steps

    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            line_str = line.strip()
            if not line_str:
                continue

            with _telemetry_lock:
                _telemetry_state["logs"].append(line_str)
                if len(_telemetry_state["logs"]) > 300:
                    _telemetry_state["logs"].pop(0)

                # Parse training step
                m_step = step_pattern.search(line_str)
                if m_step:
                    _telemetry_state["step"] = int(m_step.group(1))
                    _telemetry_state["train_loss"] = float(m_step.group(2))
                    _telemetry_state["tokens_processed"] = int(
                        m_step.group(4).replace(",", "")
                    )

                # Parse evaluation step
                m_eval = eval_pattern.search(line_str)
                if m_eval:
                    _telemetry_state["val_loss"] = float(m_eval.group(2))
                    _telemetry_state["perplexity"] = float(m_eval.group(3))

        proc.stdout.close()

    proc.wait()
    with _telemetry_lock:
        _telemetry_state["running"] = False
        _telemetry_state["logs"].append(
            f"[Training Finished] Process exited with code {proc.returncode}"
        )


@router.post("/train/start")
async def start_training(body: TrainRequest, request: Request):
    global _telemetry_state

    if request.app.state.train_process is not None:
        poll = request.app.state.train_process.poll()
        if poll is None:
            return {"status": "already_running"}

    cmd = [sys.executable, str(_PROJECT_ROOT / "training" / "train.py")]

    max_steps = 5 if body.mode == "smoke_test" else (body.steps or (body.epochs * 1000))

    if body.mode == "smoke_test":
        cmd.append("--smoke-test")
    else:
        cmd.extend([
            "--epochs", str(body.epochs),
            "--batch-size", str(body.batch_size),
            "--lr", str(body.learning_rate),
        ])
        if body.steps:
            cmd.extend(["--steps", str(body.steps)])

    # Reset telemetry
    with _telemetry_lock:
        _telemetry_state = {
            "running": True,
            "step": 0,
            "max_steps": max_steps,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "perplexity": 0.0,
            "tokens_processed": 0,
            "logs": [f"[Training] Initiated command: {' '.join(cmd)}"],
        }

    process = subprocess.Popen(
        cmd,
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    request.app.state.train_process = process

    # Launch background reader thread
    t = threading.Thread(
        target=_stdout_reader_worker,
        args=(process, max_steps),
        daemon=True,
    )
    t.start()

    return {"status": "started", "pid": process.pid, "mode": body.mode}


@router.get("/train/status", response_model=TrainStatusResponse)
async def train_status(request: Request):
    proc = request.app.state.train_process
    is_alive = False
    if proc is not None:
        poll = proc.poll()
        is_alive = (poll is None)

    with _telemetry_lock:
        return TrainStatusResponse(
            running=is_alive,
            step=_telemetry_state["step"],
            max_steps=_telemetry_state["max_steps"],
            train_loss=_telemetry_state["train_loss"],
            val_loss=_telemetry_state["val_loss"],
            perplexity=_telemetry_state["perplexity"],
            tokens_processed=_telemetry_state["tokens_processed"],
            logs=list(_telemetry_state["logs"]),
        )


@router.post("/train/stop")
async def stop_training(request: Request):
    proc = request.app.state.train_process
    if proc is None:
        return {"status": "not_running"}

    proc.terminate()
    request.app.state.train_process = None
    with _telemetry_lock:
        _telemetry_state["running"] = False
        _telemetry_state["logs"].append("[Training] Stopped by user command.")
    return {"status": "stopped"}
