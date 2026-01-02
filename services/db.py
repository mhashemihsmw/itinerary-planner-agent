"""PostgreSQL helpers using psycopg2 for video + itinerary storage."""

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import psycopg2
from psycopg2.extras import RealDictCursor

log = logging.getLogger(__name__)

DB_SCHEMA = os.getenv("DB_SCHEMA", "itinerary")


def _get_dsn() -> str:
    """Build the DSN string from env variables or DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    parts = {
        "dbname": os.getenv("PGDATABASE"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
        "host": os.getenv("PGHOST"),
        "port": os.getenv("PGPORT"),
    }
    return " ".join(f"{k}={v}" for k, v in parts.items())


@contextmanager
def get_conn():
    """Context manager yielding a connection with schema set."""
    conn = psycopg2.connect(dsn=_get_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {DB_SCHEMA};")
        yield conn
    finally:
        conn.close()


def parse_video_identity(video_url: str) -> Tuple[str, str]:
    """Derive platform + video_id from a URL with sensible fallbacks."""
    parsed = urlparse(video_url)
    host = (parsed.hostname or "unknown").lower()
    platform = host.replace("www.", "")

    video_id: Optional[str] = None
    if "youtube" in host or "youtu.be" in host:
        if parsed.hostname == "youtu.be":
            video_id = parsed.path.strip("/")
        else:
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
            if not video_id:
                parts = [p for p in parsed.path.split("/") if p]
                if parts:
                    video_id = parts[-1]
    elif "tiktok" in host or "instagram" in host:
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            video_id = parts[-1]

    if not video_id:
        video_id = hashlib.sha256(video_url.encode("utf-8")).hexdigest()

    return platform, video_id


def get_video(platform: str, video_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a video row if it exists."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, platform, video_id, video_url, transcription, created_at
                FROM videos
                WHERE platform = %s AND video_id = %s
                LIMIT 1
                """,
                (platform, video_id),
            )
            row = cur.fetchone()
    return row


def upsert_video_with_transcription(
    platform: str, video_id: str, video_url: str, transcription: str
) -> Dict[str, Any]:
    """Insert or update a video row, preserving existing transcription when present."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO videos (platform, video_id, video_url, transcription)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (platform, video_id)
                DO UPDATE SET
                    video_url = EXCLUDED.video_url,
                    transcription = COALESCE(videos.transcription, EXCLUDED.transcription)
                RETURNING id, platform, video_id, video_url, transcription, created_at
                """,
                (platform, video_id, video_url, transcription),
            )
            row = cur.fetchone()
        conn.commit()
    return row


def insert_itinerary(
    video_id: int,
    itinerary_json: Any,
    preferences: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert a new itinerary version for a video, storing user preferences."""
    preferences = preferences or {}
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT COALESCE(MAX(version), 0) AS current_version
                FROM itineraries
                WHERE video_id = %s
                """,
                (video_id,),
            )
            current = cur.fetchone()["current_version"]
            next_version = current + 1

            cur.execute(
                """
                INSERT INTO itineraries (
                    video_id,
                    itinerary_json,
                    version,
                    start_date,
                    end_date,
                    flexible_month,
                    flexible_days,
                    budget,
                    pace,
                    food_preference,
                    extra_notes
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, video_id, itinerary_json, version, created_at,
                          start_date, end_date, flexible_month, flexible_days,
                          budget, pace, food_preference, extra_notes
                """,
                (
                    video_id,
                    json.dumps(itinerary_json),
                    next_version,
                    preferences.get("start_date"),
                    preferences.get("end_date"),
                    preferences.get("flexible_month"),
                    preferences.get("flexible_days"),
                    preferences.get("budget"),
                    preferences.get("pace"),
                    preferences.get("food_preference"),
                    preferences.get("extra_notes"),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return row


def get_latest_itinerary(video_id: int) -> Optional[Dict[str, Any]]:
    """Return the most recent itinerary for a given video."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, video_id, itinerary_json, version, created_at,
                       start_date, end_date, flexible_month, flexible_days,
                       budget, pace, food_preference, extra_notes
                FROM itineraries
                WHERE video_id = %s
                ORDER BY version DESC
                LIMIT 1
                """,
                (video_id,),
            )
            row = cur.fetchone()
    return row


def list_videos(limit: int = 100) -> list[Dict[str, Any]]:
    """Return recent videos for dashboard display."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    v.id,
                    v.platform,
                    v.video_id,
                    v.video_url,
                    v.transcription,
                    v.created_at,
                    i.itinerary_json,
                    i.version AS itinerary_version,
                    i.created_at AS itinerary_created_at,
                    i.start_date,
                    i.end_date,
                    i.flexible_month,
                    i.flexible_days,
                    i.budget,
                    i.pace,
                    i.food_preference,
                    i.extra_notes
                FROM videos v
                LEFT JOIN LATERAL (
                    SELECT itinerary_json, version, created_at,
                           start_date, end_date, flexible_month, flexible_days,
                           budget, pace, food_preference, extra_notes
                    FROM itineraries
                    WHERE video_id = v.id
                    ORDER BY version DESC
                    LIMIT 1
                ) i ON TRUE
                ORDER BY v.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return rows
