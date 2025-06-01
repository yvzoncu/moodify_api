# Moodify API

A FastAPI-based backend service for song recommendations and playlist management based on mood and musical preferences.

## Features

- Song recommendations based on mood and preferences using OpenAI embeddings
- Playlist management (create, update, delete)
- Vector similarity search for song matching
- PostgreSQL with pgvector for efficient vector operations

## Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- OpenAI API key

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yvzoncu/moodify_api.git
cd moodify_api
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install fastapi uvicorn openai psycopg pgvector python-dotenv
```

4. Create `.env` file with your configuration:

```
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
OPENAI_API_KEY=your_openai_api_key
```

5. Run the application:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /api/new-song-suggester`: Get song recommendations based on query
- `GET /api/get-user-playlist`: Get user's playlists
- `POST /api/create-user-playlist`: Create a new playlist
- `POST /api/update-user-playlist`: Add/remove songs from playlist
- `DELETE /api/delete-user-playlist`: Delete a playlist

## Database Setup

Make sure to have the pgvector extension installed in your PostgreSQL database:

```sql
CREATE EXTENSION vector;

CREATE TABLE song_data (
    id SERIAL PRIMARY KEY,
    song VARCHAR NOT NULL,
    artist VARCHAR NOT NULL,
    song_info TEXT,
    genre VARCHAR,
    tempo FLOAT,
    danceability FLOAT,
    energy FLOAT,
    acousticness FLOAT,
    valence FLOAT,
    release_year INTEGER,
    embeddings vector(1536)
);

CREATE INDEX ON song_data USING hnsw (embeddings vector_cosine_ops);
```
