
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':

    print("[Train] Downloading football dataset from Roboflow...")
    rf = Roboflow(api_key="wobKTsvwpkGfnouTdxLe")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    dataset = project.version(1).download("yolov8")
    dataset_path = Path(dataset.location) / "data.yaml"
    print(f"[Train] Dataset saved to: {dataset.location}")

    print("[Train] Loading YOLOv8 base model...")
    model = YOLO("yolov8n.pt")

    print("[Train] Starting fine-tuning... this takes 20-40 minutes on your GPU")
    model.train(
        data    = str(dataset_path),
        epochs  = 50,
        imgsz   = 640,
        batch   = 8,
        device  = "cuda",
        name    = "football_ball",
        project = "runs/train",
        patience= 10,
        save    = True,
        plots   = True,
        workers = 0,
    )

    print("\n[Train] Done!")
    print("[Train] Your model is saved at: runs/train/football_ball/weights/best.pt")
    print("[Train] Now update run.py: change model_path to 'runs/train/football_ball/weights/best.pt'")
