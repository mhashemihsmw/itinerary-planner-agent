-- Database schema for the itinerary planner.
-- Matches the columns used in services/db.py. Safe to re-run.

CREATE SCHEMA IF NOT EXISTS itinerary;

CREATE TABLE IF NOT EXISTS itinerary.videos (
    id SERIAL PRIMARY KEY,
    platform TEXT NOT NULL,
    video_id TEXT NOT NULL,
    video_url TEXT NOT NULL,
    transcription TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_video UNIQUE (platform, video_id)
);

CREATE TABLE IF NOT EXISTS itinerary.itineraries (
    id SERIAL PRIMARY KEY,
    video_id INT NOT NULL REFERENCES itinerary.videos(id) ON DELETE CASCADE,
    itinerary_json JSONB NOT NULL,
    version INT NOT NULL,
    start_date DATE,
    end_date DATE,
    flexible_month TEXT,
    flexible_days INT,
    budget TEXT,
    pace TEXT,
    food_preference TEXT,
    extra_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_itineraries_video_version
    ON itinerary.itineraries(video_id, version DESC);
