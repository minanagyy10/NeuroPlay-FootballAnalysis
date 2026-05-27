"""
backend/main.py
───────────────
FastAPI backend for the Football AI system.

Endpoints
---------
POST /analyse          — Upload video, run full pipeline, return job ID
GET  /status/{job_id}  — Poll processing status
GET  /report/{job_id}  — Get JSON analysis report
GET  /video/{job_id}   — Stream annotated video
GET  /heatmap/{job_id}/{player_id} — Get player heatmap image
GET  /health           — Health check
"""

from __future__ import annotations
import asyncio
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import FootballPipeline, PipelineConfig


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football AI Analysis API",
    description="Upload a football match video → AI analyses everything.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Job store (in-memory; use Redis/DB for production) ───────────────────────

class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}

    def create(self, job_id: str) -> None:
        self._jobs[job_id] = {
            "id":      job_id,
            "status":  "queued",
            "progress": 0,
            "error":   None,
            "report":  None,
        }

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)


jobs = JobStore()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Football AI Analysis API"}


@app.post("/analyse")
async def analyse_video(
    background_tasks: BackgroundTasks,
    video:            UploadFile = File(...),
    model_path:       str        = Form("yolov8x.pt"),
    device:           str        = Form("cuda"),
    enable_tactical:  bool       = Form(True),
):
    """
    Upload a football video and start analysis.
    Returns a job_id — poll /status/{job_id} to track progress.
    """
    if not video.filename.endswith((".mp4", ".avi", ".mkv", ".mov")):
        raise HTTPException(400, "Unsupported video format. Use mp4/avi/mkv/mov.")

    job_id     = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"

    # Save uploaded file
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    jobs.create(job_id)

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline,
        job_id     = job_id,
        video_path = str(video_path),
        model_path = model_path,
        device     = device,
        enable_tactical = enable_tactical,
    )

    return JSONResponse({
        "job_id":  job_id,
        "status":  "queued",
        "message": "Analysis started. Poll /status/{job_id} for updates.",
    })


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return JSONResponse(job)


@app.get("/report/{job_id}")
async def get_report(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    if job["status"] != "complete":
        raise HTTPException(400, f"Job status is '{job['status']}', not complete.")
    return JSONResponse(job["report"])


@app.get("/video/{job_id}")
async def get_annotated_video(job_id: str):
    path = OUTPUT_DIR / job_id / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Annotated video not ready yet.")
    return FileResponse(str(path), media_type="video/mp4")


@app.get("/heatmap/{job_id}/{player_id}")
async def get_heatmap(job_id: str, player_id: int):
    path = OUTPUT_DIR / job_id / "heatmaps" / f"player_{player_id}_heatmap.jpg"
    if not path.exists():
        raise HTTPException(404, f"Heatmap for player {player_id} not found.")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/heatmap/{job_id}/team")
async def get_team_heatmap(job_id: str):
    path = OUTPUT_DIR / job_id / "heatmaps" / "team_heatmap.jpg"
    if not path.exists():
        raise HTTPException(404, "Team heatmap not ready yet.")
    return FileResponse(str(path), media_type="image/jpeg")


# ─── Background task ─────────────────────────────────────────────────────────

def _run_pipeline(
    job_id:          str,
    video_path:      str,
    model_path:      str,
    device:          str,
    enable_tactical: bool,
) -> None:
    jobs.update(job_id, status="processing", progress=5)

    out_dir = OUTPUT_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cfg = PipelineConfig(
            video_path       = video_path,
            output_path      = str(out_dir / "annotated.mp4"),
            model_path       = model_path,
            device           = device,
            enable_tactical  = enable_tactical,
            heatmap_dir      = str(out_dir / "heatmaps"),
            report_path      = str(out_dir / "report.json"),
        )

        jobs.update(job_id, progress=10)
        pipeline = FootballPipeline(cfg)

        jobs.update(job_id, progress=15)
        report = pipeline.run()

        jobs.update(job_id, status="complete", progress=100, report=report)

    except Exception as e:
        jobs.update(job_id, status="failed", error=str(e))
        raise


# ─── Dev server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)