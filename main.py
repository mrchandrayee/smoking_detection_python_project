#!/usr/bin/env python3
"""Smoking detection CCTV pipeline reconstructed from the supplied screen recording."""
import argparse
from pathlib import Path
from smoking_detection.pipeline import SmokingDetectionPipeline

def parse_args():
    p = argparse.ArgumentParser(description="CCTV smoking detection with person crops and temporal alerts")
    p.add_argument("--source", default="0", help="Camera index or video path")
    p.add_argument("--person-weights", default="yolo26s.pt", help="YOLO person detector weights")
    p.add_argument("--smoking-weights", default="smoking_detection/best_smoke.pt",
                   help="Custom smoking/cigarette detector weights")
    p.add_argument("--config", default="config/settings.json")
    p.add_argument("--output", default="output/smoking_detection.mp4")
    p.add_argument("--display", action="store_true")
    p.add_argument("--save-snapshots", action="store_true")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or CUDA index")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SmokingDetectionPipeline(args).run()
