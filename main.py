"""
AI Trip Planner Core - FastAPI MVP
A backend service that extracts travel information and generates structured itineraries
"""
import asyncio
import logging
import os
import tempfile
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL

from models.schemas import UserPreferences
from services import db as db_service
from services.openai_service import OpenAIService

# Load environment variables
load_dotenv()

log = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="AI Trip Planner Core",
    description="Extract travel information and generate structured itineraries",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI service
openai_service = OpenAIService(api_key=os.getenv("OPENAI_API_KEY"))

# Lazy Whisper model (loaded on first use)
_whisper_model: Optional[WhisperModel] = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        model_name = os.getenv("WHISPER_MODEL", "small")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        _whisper_model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
        )
    return _whisper_model


# Request/Response Models
class ItineraryRequest(BaseModel):
    video_url: str = Field(..., description="URL of the travel video (YouTube/Instagram/TikTok or any yt-dlp supported link)")
    caption_text: Optional[str] = Field(
        None,
        description="Optional caption text; auto-fetch uses yt-dlp + Whisper for supported platforms",
    )
    start_date: Optional[date] = Field(None, description="Trip start date (YYYY-MM-DD) if fixed")
    end_date: Optional[date] = Field(None, description="Trip end date (YYYY-MM-DD) if fixed")
    flexible_month: Optional[str] = Field(None, description="Month window (YYYY-MM) if dates are flexible")
    flexible_days: Optional[int] = Field(None, description="Number of days within the flexible month", ge=1, le=60)
    budget: Optional[str] = Field(None, description="value | balanced | luxury")
    pace: Optional[str] = Field(None, description="relaxed | active")
    food_preference: Optional[str] = Field(None, description="vegan | vegetarian | all-eater")
    extra_notes: Optional[str] = Field(None, description="Free-text notes (start time, nightlife, etc.)", max_length=800)

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://www.youtube.com/shorts/iYYyIFRTq6g",
                "caption_text": "Trip to Oslo.",
                "start_date": "2025-05-10",
                "end_date": "2025-05-15",
                "budget": "balanced",
                "pace": "relaxed",
                "food_preference": "all-eater",
                "extra_notes": "We start days at 9am, love local coffee and mild nightlife."
            }
        }


class DailyActivity(BaseModel):
    day: int
    title: str
    activities: List[str]


class ItineraryResponse(BaseModel):
    destination: str
    best_season: str
    duration_days: int
    daily_plan: List[DailyActivity]
    key_attractions: List[str]
    food_suggestions: List[str]



async def fetch_caption_text(video_url: str) -> str:
    """Fetch caption text via yt-dlp + Whisper; works for any yt-dlp supported URL."""

    def _download_and_transcribe() -> str:
        # Download audio with yt-dlp then run Whisper locally
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outtmpl = str(tmp_path / "%(id)s.%(ext)s")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            audio_files = list(tmp_path.glob("*.mp3"))
            if not audio_files:
                raise HTTPException(status_code=400, detail="Failed to download audio for transcription")

            audio_file = audio_files[0]
            model = get_whisper_model()
            segments, _info = model.transcribe(
                str(audio_file),
                beam_size=1,
                temperature=0,
                vad_filter=True,
            )
            text_parts = [seg.text.strip() for seg in segments if seg.text]
            caption_text = " ".join(text_parts).strip()
            log.info("whisper transcription length=%s sample=%s", len(caption_text), caption_text[:200])
            if not caption_text:
                raise HTTPException(status_code=400, detail="Transcription produced empty text")
            return caption_text

    def _fetch_metadata_text() -> str:
        """Fetch video title/description without downloading media."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        parts = []
        for key in ("title", "description"):
            val = info.get(key)
            if val:
                parts.append(str(val).strip())
        metadata_text = " \n".join(parts).strip()
        log.info("metadata length=%s sample=%s", len(metadata_text), metadata_text[:200])
        return metadata_text

    try:
        caption_text = await asyncio.to_thread(_download_and_transcribe)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to fetch captions")
        raise HTTPException(status_code=400, detail=f"Could not fetch captions for video: {str(e)}")

    # Try to enrich with title/description even if transcription succeeded
    metadata_text = ""
    try:
        metadata_text = await asyncio.to_thread(_fetch_metadata_text)
    except Exception:
        log.warning("metadata fetch failed for %s", video_url)

    combined = " ".join(part for part in (caption_text, metadata_text) if part).strip()
    log.info("combined caption length=%s sample=%s", len(combined), combined[:200])
    if len(combined) < 10:
        raise HTTPException(status_code=400, detail="Caption text is too short; provide caption_text manually or use a video with audio/description.")
    return combined


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "AI Trip Planner Core",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/videos")
async def list_videos(limit: int = 100):
    """List recent videos for the dashboard."""
    videos = await asyncio.to_thread(db_service.list_videos, limit)
    return {"items": videos}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple HTML dashboard for submitting video links and viewing history."""
    try:
        html = load_dashboard_html()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Dashboard template missing")
    return HTMLResponse(content=html)


@lru_cache(maxsize=1)
def load_dashboard_html() -> str:
    """Load the dashboard HTML from the templates directory (cached)."""
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/videos/{video_id}/itinerary")
async def get_video_itinerary(video_id: int):
    """Fetch the latest itinerary for a specific video."""
    itinerary = await asyncio.to_thread(db_service.get_latest_itinerary, video_id)
    if not itinerary:
        raise HTTPException(status_code=404, detail="Itinerary not found for this video")
    return itinerary


@app.post("/generate-itinerary", response_model=ItineraryResponse)
async def generate_itinerary(request: ItineraryRequest):
    """
    Generate a structured travel itinerary from video caption text.
    
    This endpoint:
    1. Fetches captions automatically if none provided
    2. Extracts travel facts from the caption text
    3. Generates a day-by-day itinerary
    4. Returns structured JSON
    """
    try:
        platform, external_id = db_service.parse_video_identity(request.video_url)

        # Validate date inputs and enforce a choice between fixed vs flexible
        fixed_dates = bool(request.start_date and request.end_date)
        flexible_window = bool(request.flexible_month and request.flexible_days)

        if request.start_date and request.end_date and request.start_date > request.end_date:
            raise HTTPException(status_code=400, detail="start_date cannot be after end_date")
        if (request.start_date and not request.end_date) or (request.end_date and not request.start_date):
            raise HTTPException(status_code=400, detail="Provide both start_date and end_date or neither")
        if not fixed_dates and not flexible_window:
            raise HTTPException(status_code=400, detail="Choose either fixed dates or a flexible month with number of days")

        preferences = UserPreferences(
            start_date=request.start_date,
            end_date=request.end_date,
            flexible_month=request.flexible_month,
            flexible_days=request.flexible_days,
            budget=request.budget,
            pace=request.pace,
            food_preference=request.food_preference,
            extra_notes=request.extra_notes,
        )

        caption_text = (request.caption_text or "").strip()

        # Prefer cached transcription if available
        if not caption_text:
            cached_video = await asyncio.to_thread(db_service.get_video, platform, external_id)
            if cached_video and cached_video.get("transcription"):
                caption_text = cached_video["transcription"]
            else:
                caption_text = await fetch_caption_text(request.video_url)

        # Ensure the video row exists with transcription persisted
        video_row = await asyncio.to_thread(
            db_service.upsert_video_with_transcription,
            platform,
            external_id,
            request.video_url,
            caption_text,
        )

        # Generate itinerary using two-step AI pipeline with user overrides
        itinerary = await openai_service.generate_itinerary(
            caption_text=caption_text,
            video_url=request.video_url,
            preferences=preferences,
        )

        # If user dates or flexible days are provided, enforce duration override
        if preferences.start_date and preferences.end_date:
            itinerary.duration_days = (preferences.end_date - preferences.start_date).days + 1
            itinerary.start_date = preferences.start_date
            itinerary.end_date = preferences.end_date
        elif preferences.flexible_days:
            itinerary.duration_days = preferences.flexible_days
            itinerary.flexible_month = preferences.flexible_month
            itinerary.flexible_days = preferences.flexible_days
        else:
            itinerary.flexible_month = preferences.flexible_month

        # Persist preferences back onto itinerary for transparency
        itinerary.budget = preferences.budget
        itinerary.pace = preferences.pace
        itinerary.food_preference = preferences.food_preference
        itinerary.extra_notes = preferences.extra_notes

        # Store itinerary versioned in the DB
        await asyncio.to_thread(
            db_service.insert_itinerary,
            video_row["id"],
            itinerary.model_dump(),
            preferences.model_dump(exclude_none=False),
        )

        return itinerary
    
    except HTTPException as e:
        # Preserve HTTPExceptions from validation/caption fetch
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate itinerary: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
