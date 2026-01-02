# AI Trip Planner Core

FastAPI backend that transcribes travel videos, extracts travel facts, and generates structured itineraries with OpenAI. Captions are cached, itineraries are versioned in PostgreSQL, and a minimal dashboard lists recent runs.

## Features

- Caption ingestion via yt-dlp + faster-whisper (with title/description fallback)
- Two-step OpenAI pipeline (fact extraction → itinerary generation)
- PostgreSQL persistence with versioned itineraries and cached transcriptions
- Simple HTML dashboard to submit links and view history
- Async FastAPI API with CORS enabled

## Prerequisites

- Python 3.9+
- FFmpeg available on PATH (required for yt-dlp audio extraction)
- PostgreSQL database (schema `itinerary` by default)
- OpenAI API key

## Setup

```bash
cd "d:\My Files\Projects\Apps\Voya\info-extractor"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create `.env` in the project root:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
DB_SCHEMA=itinerary                  # optional, default: itinerary
WHISPER_MODEL=small                  # optional (see faster-whisper docs)
WHISPER_COMPUTE_TYPE=int8            # optional: int8, float16, etc.
```

### Database schema (SQL)

Run once in your database (adjust schema name if you changed `DB_SCHEMA`):

```sql
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_itineraries_video_version
    ON itinerary.itineraries(video_id, version DESC);
```

## Run the API

```bash
uvicorn main:app --reload
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Dashboard: http://localhost:8000/dashboard

## Endpoints

- `GET /` – health check
- `GET /videos` – list recent videos with latest itinerary
- `GET /videos/{id}/itinerary` – latest itinerary for a video row
- `GET /dashboard` – lightweight HTML dashboard
- `POST /generate-itinerary` – generate and store itinerary (see payload below)

Request payload:

```json
{
  "video_url": "https://www.youtube.com/watch?v=dummy",
  "caption_text": "Optional override; if omitted we auto-transcribe"
}
```

## How it works

1) Fetch captions
- If `caption_text` is missing, download audio with yt-dlp and transcribe with faster-whisper; enrich with title/description metadata.
- Transcription is cached per `(platform, video_id)` so repeated calls avoid re-transcribing.

2) AI pipeline
- Step 1: `gpt-4o-mini` extracts structured travel facts (low temperature).
- Step 2: `gpt-4o-mini` produces a paced day-by-day itinerary (moderate temperature).
- Responses are validated with Pydantic models in [models/schemas.py](models/schemas.py).

3) Persistence
- Videos and their transcriptions are upserted; itineraries are stored as versioned JSON in PostgreSQL for history.

## Testing

```bash
pytest
```

CI-style quick check for transcription logic only:

```bash
pytest tests/test_fetch_caption_text.py
```

## Troubleshooting

- Whisper model download happens on first run; expect a pause and network usage.
- Ensure FFmpeg is on PATH; yt-dlp/ffmpeg errors often indicate a missing binary.
- If database errors appear, verify `DATABASE_URL`/`PG*` env vars and that the schema/tables exist.

## License

MIT
