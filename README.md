# AI Trip Planner Core

Turn a travel video into a ready-to-use trip plan.

Paste a link to a travel video from YouTube, TikTok or Instagram, and tell the app when you're travelling and what you like (budget, pace, food). The app listens to the video, picks out the places and tips it mentions, and builds a day-by-day itinerary that fits your dates and preferences. Every plan is saved, so you can revisit past videos and their itineraries from a simple dashboard.

## Getting started

### Requirements

- Python 3.12 (tested). Python 3.14 does not work with the pinned dependencies.
- FFmpeg (on Ubuntu/WSL: `sudo apt install -y ffmpeg`)
- Docker, for the PostgreSQL database
- An OpenAI API key

Commands are for Linux, WSL and macOS. On Windows PowerShell, activate the venv with `.venv\Scripts\activate`.

### 1. Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure `.env`

Create `.env` in the project root. It is git-ignored, so never commit it.

```
OPENAI_API_KEY=sk-...

PGDATABASE=Voya
PGUSER=postgres
PGPASSWORD=password
PGHOST=localhost
PGPORT=5432
```

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | none | Full connection string; overrides the `PG*` variables |
| `DB_SCHEMA` | `itinerary` | Database schema name |
| `WHISPER_MODEL` | `small` | Speech-to-text model size |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` for CPU, `float16` for GPU |
| `HF_TOKEN` | none | Faster speech-to-text model downloads |

### 3. Start the database

The user, password and database name must match `.env`.

```bash
docker run -d --name voya-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=Voya \
  -p 127.0.0.1:5432:5432 \
  -v voya-pgdata:/var/lib/postgresql/data \
  postgres:16

docker exec -i voya-pg psql -U postgres -d Voya < schema.sql
```

Data is stored in the `voya-pgdata` volume, so it survives restarts. You can also browse the data in DBeaver or pgAdmin: connect to `localhost:5432`, database `Voya`, using the credentials from `.env`. The tables are in the `itinerary` schema.

### 4. Run and verify

```bash
pytest tests/              # all tests should pass
uvicorn main:app --reload
```

Open http://localhost:8000/dashboard and submit a short travel video. The first run is slow because the speech-to-text model is downloaded.

### Daily workflow

```bash
source .venv/bin/activate
docker start voya-pg
uvicorn main:app --reload
```

To reset the database, run `docker rm -f voya-pg && docker volume rm voya-pgdata`, then repeat step 3.

## API

Interactive docs: http://localhost:8000/docs

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/dashboard` | Web dashboard |
| `GET` | `/videos` | Recent videos with their latest itinerary |
| `GET` | `/videos/{id}/itinerary` | Latest itinerary for a video |
| `POST` | `/generate-itinerary` | Generate and save an itinerary |

`POST /generate-itinerary` needs **either** `start_date` + `end_date` **or** `flexible_month` + `flexible_days`:

```json
{
  "video_url": "https://www.youtube.com/watch?v=RiJhQkvlCok",
  "start_date": "2026-12-01",
  "end_date": "2026-12-05",
  "budget": "balanced",
  "pace": "relaxed",
  "food_preference": "all-eater",
  "extra_notes": "We start days at 9am and like local coffee."
}
```

| Field | Values |
| --- | --- |
| `video_url` | Any link yt-dlp supports |
| `caption_text` | Optional; skips transcription when provided |
| `start_date`, `end_date` | `YYYY-MM-DD` |
| `flexible_month`, `flexible_days` | `YYYY-MM`, 1–60 |
| `budget` | `value`, `balanced`, `luxury` |
| `pace` | `relaxed`, `active` |
| `food_preference` | `vegan`, `vegetarian`, `all-eater` |
| `extra_notes` | Free text, max 800 characters |

## How it works

1. **Transcribe**: yt-dlp downloads the audio, faster-whisper transcribes it locally, and the video's title and description are added. Transcripts are cached per video, so each video is only transcribed once.
2. **Extract facts**: `gpt-4o-mini` pulls structured travel facts out of the transcript.
3. **Plan**: `gpt-4o-mini` turns those facts and your preferences into a day-by-day itinerary, validated with Pydantic.
4. **Save**: each itinerary is stored in PostgreSQL as a new version for that video.

## Project layout

```
main.py                        API routes, request validation, video transcription
models/schemas.py              Pydantic models (preferences, travel facts, itinerary)
services/openai_service.py     Two-step OpenAI pipeline
services/db.py                 PostgreSQL access
services/gemeni_service.py     Experimental Gemini pipeline (not used by the app)
services/langchain_service.py  Experimental LangChain pipeline (not used by the app)
templates/dashboard.html       Dashboard page
schema.sql                     Database schema
tests/                         Unit tests
test_api.py                    Manual smoke test; needs a running server
```

## Development guidelines

- **Schema changes**: update `schema.sql` and `services/db.py` together. There is no migration tool, so apply `ALTER TABLE` statements to existing databases by hand.
- **Models**: keep shared models in `models/schemas.py`; don't duplicate them in `main.py`.
- **JSON columns**: store Pydantic models with `model_dump(mode="json")`, because plain `model_dump()` keeps `date` objects that can't be serialized.
- **Blocking calls**: database, yt-dlp and Whisper calls are synchronous. Wrap them in `asyncio.to_thread(...)` inside async routes.
- **Dependencies**: after adding or upgrading a package, run `pip check` and `pytest tests/`, then update its pin in `requirements.txt`.
- **Tests**: put tests in `tests/` and mock yt-dlp, Whisper and OpenAI so tests run offline. Run `pytest tests/` rather than plain `pytest`, which also picks up `test_api.py`.
- **Secrets**: keep them in `.env` only.

## License

MIT
