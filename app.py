import os
import uuid
import base64
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="MoveScan Video Frame Extractor")


# ---------------------------------------------------------
# CORS
# Allows the MoveScan survey website to call Render
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://videosurvey.autocloserhq.com",
        "https://autocloserhq.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class ExtractRequest(BaseModel):
    video_url: str
    frame_count: Optional[int] = 5


class SubmittedVideo(BaseModel):
    room_key: Optional[str] = None
    room_label: Optional[str] = None
    video_url: str


class VideoSubmission(BaseModel):
    client_code: Optional[str] = None
    survey_type: Optional[str] = "Video"
    name: str
    email: str
    phone: str
    move_date: Optional[str] = None
    from_zip: Optional[str] = None
    to_zip: Optional[str] = None
    videos: List[SubmittedVideo]
    submitted_at: Optional[str] = None
    page_url: Optional[str] = None


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "MoveScan Video Frame Extractor"
    }


# ---------------------------------------------------------
# MOVESCAN SUBMISSION SMS ALERT
# Does not extract frames or run OpenAI
# ---------------------------------------------------------

@app.post("/submission-alert")
def submission_alert(payload: VideoSubmission):
    if not payload.videos:
        raise HTTPException(
            status_code=400,
            detail="At least one video is required"
        )

    try:
        telnyx_response = send_submission_text(payload)

        return {
            "success": True,
            "message": "MoveScan submission text alert sent",
            "customer": payload.name,
            "video_count": len(payload.videos),
            "telnyx_message_id": telnyx_response.get(
                "data",
                {}
            ).get("id")
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Submission alert error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Could not send MoveScan submission text alert"
        )


def send_submission_text(payload: VideoSubmission):
    telnyx_api_key = os.getenv("TELNYX_API_KEY")
    telnyx_from_number = os.getenv("TELNYX_FROM_NUMBER")
    alert_phone_number = os.getenv("ALERT_PHONE_NUMBER")

    if not telnyx_api_key:
        raise RuntimeError(
            "TELNYX_API_KEY environment variable is missing"
        )

    if not telnyx_from_number:
        raise RuntimeError(
            "TELNYX_FROM_NUMBER environment variable is missing"
        )

    if not alert_phone_number:
        raise RuntimeError(
            "ALERT_PHONE_NUMBER environment variable is missing"
        )

    company_name = get_company_name(payload.client_code)
    video_count = len(payload.videos)

    move_date = payload.move_date or "Not provided"
    from_zip = payload.from_zip or "N/A"
    to_zip = payload.to_zip or "N/A"

    text_message = (
        f"New MoveScan submitted.\n"
        f"Customer: {payload.name}\n"
        f"Company: {company_name}\n"
        f"Videos: {video_count}\n"
        f"Route: {from_zip} to {to_zip}\n"
        f"Move date: {move_date}\n"
        f"Open Make and run the queued Video MoveScan workflow."
    )

    response = requests.post(
        "https://api.telnyx.com/v2/messages",
        headers={
            "Authorization": f"Bearer {telnyx_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": normalize_phone_number(telnyx_from_number),
            "to": normalize_phone_number(alert_phone_number),
            "text": text_message,
        },
        timeout=30,
    )

    if response.status_code not in (200, 201, 202):
        print(
            "Telnyx SMS error:",
            response.status_code,
            response.text
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Telnyx rejected the SMS request. "
                f"Status: {response.status_code}. "
                f"Response: {response.text[:500]}"
            )
        )

    return response.json()


def normalize_phone_number(phone_number: str) -> str:
    cleaned = (
        phone_number
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )

    if cleaned.startswith("+"):
        return cleaned

    if len(cleaned) == 10:
        return f"+1{cleaned}"

    if len(cleaned) == 11 and cleaned.startswith("1"):
        return f"+{cleaned}"

    raise ValueError(
        f"Invalid phone number format: {phone_number}"
    )


def get_company_name(client_code: Optional[str]) -> str:
    normalized_code = (
        client_code or "default"
    ).strip().lower()

    company_names = {
        "modelmoving": "Model Moving",
        "aplus": "A Plus Moving",
        "default": "AutoCloser MoveScan",
    }

    return company_names.get(
        normalized_code,
        "AutoCloser MoveScan"
    )


# ---------------------------------------------------------
# VIDEO FRAME EXTRACTION
# Existing MoveScan functionality
# ---------------------------------------------------------

@app.post("/extract-frames")
def extract_frames(payload: ExtractRequest):
    if not payload.video_url:
        raise HTTPException(
            status_code=400,
            detail="video_url is required"
        )

    frame_count = payload.frame_count or 5

    if frame_count < 1 or frame_count > 8:
        raise HTTPException(
            status_code=400,
            detail="frame_count must be between 1 and 8"
        )

    job_id = str(uuid.uuid4())
    work_dir = Path(f"/tmp/{job_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "input_video.mp4"

    try:
        download_video(
            payload.video_url,
            video_path
        )

        duration = get_video_duration(
            video_path
        )

        if duration <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not read video duration"
            )

        if duration > 30:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Video is too long. "
                    "Max allowed is 30 seconds."
                )
            )

        timestamps = build_timestamps(
            duration,
            frame_count
        )

        frames = []

        for index, timestamp in enumerate(timestamps):
            frame_path = (
                work_dir /
                f"frame_{index + 1}.jpg"
            )

            extract_single_frame(
                video_path=video_path,
                timestamp=timestamp,
                output_path=frame_path
            )

            if not frame_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Frame {index + 1} "
                        "was not created"
                    )
                )

            image_base64 = encode_image_base64(
                frame_path
            )

            frames.append({
                "frame_number": index + 1,
                "position": get_position_label(
                    index,
                    frame_count
                ),
                "timestamp_seconds": round(
                    timestamp,
                    2
                ),
                "image_base64": (
                    "data:image/jpeg;base64,"
                    f"{image_base64}"
                )
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
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        shutil.rmtree(
            work_dir,
            ignore_errors=True
        )


def download_video(
    video_url: str,
    video_path: Path
):
    response = requests.get(
        video_url,
        stream=True,
        timeout=60
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not download video. "
                f"Status code: {response.status_code}"
            )
        )

    max_bytes = 100 * 1024 * 1024
    downloaded = 0

    with open(video_path, "wb") as file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if not chunk:
                continue

            downloaded += len(chunk)

            if downloaded > max_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Video file is too large. "
                        "Max allowed is 100 MB."
                    )
                )

            file.write(chunk)


def get_video_duration(
    video_path: Path
) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(
                f"ffprobe error: "
                f"{result.stderr}"
            )
        )

    return float(result.stdout.strip())


def build_timestamps(
    duration: float,
    frame_count: int
):
    if frame_count == 1:
        return [
            max(0.5, duration / 2)
        ]

    start = min(
        1.0,
        duration * 0.10
    )

    end = max(
        duration - 1.0,
        duration * 0.90
    )

    if frame_count == 5:
        return [
            start,
            duration * 0.25,
            duration * 0.50,
            duration * 0.75,
            end
        ]

    step = (
        end - start
    ) / (
        frame_count - 1
    )

    return [
        start + (step * index)
        for index in range(frame_count)
    ]


def extract_single_frame(
    video_path: Path,
    timestamp: float,
    output_path: Path
):
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(1024,iw)':-2",
        "-q:v",
        "3",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(
                f"ffmpeg error: "
                f"{result.stderr}"
            )
        )


def encode_image_base64(
    image_path: Path
) -> str:
    with open(
        image_path,
        "rb"
    ) as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def get_position_label(
    index: int,
    frame_count: int
) -> str:
    if frame_count == 5:
        labels = [
            "beginning",
            "25_percent",
            "50_percent",
            "75_percent",
            "end"
        ]

        return labels[index]

    return f"frame_{index + 1}"
