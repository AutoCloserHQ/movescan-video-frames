import os
import uuid
import base64
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="MoveScan Video Frame Extractor")


class ExtractRequest(BaseModel):
    video_url: str
    frame_count: Optional[int] = 5


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "MoveScan Video Frame Extractor"
    }


@app.post("/extract-frames")
def extract_frames(payload: ExtractRequest):
    if not payload.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")

    frame_count = payload.frame_count or 5

    if frame_count < 1 or frame_count > 8:
        raise HTTPException(status_code=400, detail="frame_count must be between 1 and 8")

    job_id = str(uuid.uuid4())
    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "input_video.mp4"

    try:
        # 1. Download video
        download_video(payload.video_url, video_path)

        # 2. Get video duration
        duration = get_video_duration(video_path)

        if duration <= 0:
            raise HTTPException(status_code=400, detail="Could not read video duration")

        if duration > 30:
            raise HTTPException(status_code=400, detail="Video is too long. Max allowed is 30 seconds for this test endpoint.")

        # 3. Create 5 frame timestamps
        timestamps = build_timestamps(duration, frame_count)

        # 4. Extract frames
        frames = []

        for index, timestamp in enumerate(timestamps):
            frame_path = work_dir / f"frame_{index + 1}.jpg"

            extract_single_frame(
                video_path=video_path,
                timestamp=timestamp,
                output_path=frame_path
            )

            if not frame_path.exists():
                raise HTTPException(status_code=500, detail=f"Frame {index + 1} was not created")

            image_base64 = encode_image_base64(frame_path)

            frames.append({
                "frame_number": index + 1,
                "position": get_position_label(index, frame_count),
                "timestamp_seconds": round(timestamp, 2),
                "image_base64": f"data:image/jpeg;base64,{image_base64}"
            })

        return {
            "success": True,
            "duration_seconds": round(duration, 2),
            "frame_count": len(frames),
            "frames": frames
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def download_video(video_url: str, video_path: Path):
    response = requests.get(video_url, stream=True, timeout=60)

    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"Could not download video. Status code: {response.status_code}"
        )

    max_bytes = 100 * 1024 * 1024  # 100 MB
    downloaded = 0

    with open(video_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                downloaded += len(chunk)

                if downloaded > max_bytes:
                    raise HTTPException(status_code=400, detail="Video file is too large. Max allowed is 100 MB.")

                file.write(chunk)


def get_video_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffprobe error: {result.stderr}")

    return float(result.stdout.strip())


def build_timestamps(duration: float, frame_count: int):
    if frame_count == 1:
        return [max(0.5, duration / 2)]

    # Avoid exact 0 and exact end because those can sometimes be black/blank frames.
    start = min(1.0, duration * 0.10)
    end = max(duration - 1.0, duration * 0.90)

    if frame_count == 5:
        return [
            start,
            duration * 0.25,
            duration * 0.50,
            duration * 0.75,
            end
        ]

    step = (end - start) / (frame_count - 1)
    return [start + (step * i) for i in range(frame_count)]


def extract_single_frame(video_path: Path, timestamp: float, output_path: Path):
    command = [
        "ffmpeg",
        "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "scale='min(1024,iw)':-2",
        "-q:v", "3",
        str(output_path)
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg error: {result.stderr}")


def encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_position_label(index: int, frame_count: int) -> str:
    if frame_count == 5:
        labels = ["beginning", "25_percent", "50_percent", "75_percent", "end"]
        return labels[index]

    return f"frame_{index + 1}"
