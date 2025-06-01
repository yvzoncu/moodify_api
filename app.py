from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from emotion_detector import search_songs_with_embedding
from psycopg.rows import dict_row


class PlaylistItem(BaseModel):
    song_id: int


class CreatePlaylistRequest(BaseModel):
    user_id: str
    playlist_name: str
    playlist_items: List[PlaylistItem]


load_dotenv()

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=2)

# Environment variables
DB_HOST = os.getenv("DB_HOST", "moodify.cje0wa8qioij.eu-north-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "lyricsdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Set to False
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        row_factory=dict_row,
    )


@app.get("/api/new-song-suggester")
async def search(query: str, user_id: str, k: int = 5):
    loop = asyncio.get_event_loop()

    def search_operation():
        return search_songs_with_embedding(query, top_k=k)

    songs = await loop.run_in_executor(None, search_operation)
    return {"results": songs if songs else []}


def insert_user_query(user_id, query):
    conn = get_db_connection()
    sql_query = """
        INSERT INTO user_queries (user_id, query)
        VALUES (%s, %s)
        RETURNING id;
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql_query, (user_id, query))
            query_id = cursor.fetchone()["id"]
            conn.commit()
            return query_id
    finally:
        conn.close()


# playlist ittem fetcher
def get_song_playlist_items_by_id(conn, playlist_id: int):
    result = []
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT s.*
                FROM user_playlist p
                JOIN LATERAL jsonb_array_elements(p.playlist_items) AS item ON TRUE
                JOIN songs s ON (item->>'song_id')::INT = s.id
                WHERE p.id = %s
                """,
                (playlist_id,),
            )
            songs = cursor.fetchall()
            for song in songs:
                result.append(
                    {
                        "song_id": song["id"],
                        "song": song["song"],
                        "artist": song["artist"],
                        "full_lyric": "",
                        "dominants": song["dominants"],
                        "tags": song["tags"],
                        "genre": song["genre"],
                    }
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    return result


@app.get("/api/get-user-playlist")
async def get_user_playlist(user_id: str):
    """
    Get all playlists for a specific user
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, playlist_name, playlist_items, created_at
                    FROM user_playlist
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )

                playlists = []
                for row in cursor.fetchall():
                    playlists.append(
                        {
                            "id": row["id"],
                            "user_id": row["user_id"],
                            "playlist_name": row["playlist_name"],
                            "playlist_items": row["playlist_items"],
                        }
                    )

                return {"playlists": playlists}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.delete("/api/delete-user-playlist")
async def delete_user_playlist(user_id: str, playlist_id: int):
    """
    Delete a playlist for a user and return remaining playlists
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Check if playlist exists and belongs to the user
                cursor.execute(
                    """
                    SELECT id FROM user_playlist 
                    WHERE id = %s AND user_id = %s
                    """,
                    (playlist_id, user_id),
                )

                playlist = cursor.fetchone()
                if not playlist:
                    raise HTTPException(
                        status_code=404,
                        detail="Playlist not found or doesn't belong to the user",
                    )

                # Delete the playlist
                cursor.execute(
                    "DELETE FROM user_playlist WHERE id = %s", (playlist_id,)
                )
                conn.commit()

                # Get remaining playlists for the user
                cursor.execute(
                    """
                    SELECT id, user_id, playlist_name, playlist_items
                    FROM user_playlist
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )

                playlists = []
                for row in cursor.fetchall():
                    playlists.append(
                        {
                            "id": row["id"],
                            "user_id": row["user_id"],
                            "playlist_name": row["playlist_name"],
                            "playlist_items": row["playlist_items"],
                        }
                    )

                return {"playlists": playlists}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.post("/api/create-user-playlist")
async def create_user_playlist(request: CreatePlaylistRequest):
    """
    Create a new playlist for a user
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Convert playlist items to JSON
                playlist_items_json = [
                    {"song_id": item.song_id} for item in request.playlist_items
                ]

                # Insert the new playlist
                cursor.execute(
                    """
                    INSERT INTO user_playlist (user_id, playlist_name, playlist_items)
                    VALUES (%s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        request.user_id,
                        request.playlist_name,
                        json.dumps(playlist_items_json),
                    ),
                )

                result = cursor.fetchone()
                conn.commit()

                playlist = {
                    "id": result["id"],
                    "user_id": request.user_id,
                    "playlist_name": request.playlist_name,
                    "playlist_items": request.playlist_items,
                    "created_at": (
                        result["created_at"].isoformat()
                        if result["created_at"]
                        else None
                    ),
                }

                new_playlist_items = get_song_playlist_items_by_id(conn, result["id"])

                cursor.execute(
                    """
                    SELECT id, user_id, playlist_name, playlist_items, created_at
                    FROM user_playlist
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (request.user_id,),
                )

                playlists = []
                for row in cursor.fetchall():
                    playlists.append(
                        {
                            "id": row["id"],
                            "user_id": row["user_id"],
                            "playlist_name": row["playlist_name"],
                            "playlist_items": row["playlist_items"],
                        }
                    )

                new_item = {"playlist": playlist, "items": new_playlist_items}
                return {"playlists": playlists, "new_item": new_item}

        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.post("/api/update-user-playlist")
async def update_user_playlist(playlist_id: int, song_id: int, action: str = "add"):
    """
    Add or remove a song from an existing playlist
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # First get the current playlist items
                cursor.execute(
                    "SELECT * FROM user_playlist WHERE id = %s", (playlist_id,)
                )
                playlist = cursor.fetchone()
                if not playlist:
                    raise HTTPException(status_code=404, detail="Playlist not found")

                playlist_items = playlist["playlist_items"]

                if action == "add":
                    # Check if song already exists in playlist
                    song_exists = any(
                        item.get("song_id") == song_id for item in playlist_items
                    )

                    if not song_exists:
                        # Add the new song to the playlist items
                        playlist_items.append({"song_id": song_id})

                        # Update the playlist with the new items
                        cursor.execute(
                            "UPDATE user_playlist SET playlist_items = %s::jsonb WHERE id = %s RETURNING id",
                            (json.dumps(playlist_items), playlist_id),
                        )
                        conn.commit()

                        pl = {
                            "id": playlist["id"],
                            "user_id": playlist["user_id"],
                            "playlist_name": playlist["playlist_name"],
                            "playlist_items": playlist_items,
                        }

                        items = get_song_playlist_items_by_id(conn, playlist_id)
                        return {
                            "message": "Song added to playlist",
                            "playlist": pl,
                            "items": items,
                        }
                    else:
                        return {
                            "message": "Song already exists in playlist",
                            "playlist": {},
                            "items": [],
                        }

                elif action == "remove":
                    # Filter out the song to remove
                    new_playlist_items = [
                        item
                        for item in playlist_items
                        if item.get("song_id") != song_id
                    ]

                    if len(new_playlist_items) < len(playlist_items):
                        # Update the playlist with the filtered items
                        cursor.execute(
                            "UPDATE user_playlist SET playlist_items = %s::jsonb WHERE id = %s RETURNING id",
                            (json.dumps(new_playlist_items), playlist_id),
                        )
                        conn.commit()

                        pl = {
                            "id": playlist["id"],
                            "user_id": playlist["user_id"],
                            "playlist_name": playlist["playlist_name"],
                            "playlist_items": new_playlist_items,
                        }

                        items = get_song_playlist_items_by_id(conn, playlist_id)
                        return {
                            "message": "Song removed from playlist",
                            "playlist": pl,
                            "items": items,
                        }
                    else:
                        return {
                            "message": "Song not found in playlist",
                            "playlist": {},
                            "items": [],
                        }
                else:
                    raise HTTPException(
                        status_code=400, detail="Invalid action. Use 'add' or 'remove'."
                    )

        except HTTPException as e:
            raise e
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-song-playlist-by-id")
async def get_song_playlist_by_id(id: int):
    """
    Get song by id
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, playlist_name, playlist_items
                    FROM user_playlist
                    WHERE id = %s
                    """,
                    (id,),
                )

                playlist = cursor.fetchone()
                if not playlist:
                    raise HTTPException(status_code=404, detail="Song not found")

                selected_playlist = {
                    "id": playlist["id"],
                    "user_id": playlist["user_id"],
                    "playlist_name": playlist["playlist_name"],
                    "playlist_items": playlist["playlist_items"],
                }

                items = get_song_playlist_items_by_id(conn, playlist["id"])
                return {"playlist": selected_playlist, "items": items}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-song-by-id")
async def get_song_by_id(id: int):
    """
    Get song by id
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, song, artist, full_lyric, dominants, tags, genre
                    FROM songs
                    WHERE id = %s
                    """,
                    (id,),
                )

                song = cursor.fetchone()
                if not song:
                    raise HTTPException(status_code=404, detail="Song not found")

                song_item = {
                    "song_id": song["id"],
                    "song": song["song"],
                    "artist": song["artist"],
                    "full_lyric": "",
                    "dominants": song["dominants"],
                    "tags": song["tags"],
                    "genre": song["genre"],
                }

                return song_item

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-playlist-by-playlist-id")
async def get_user_playlist(id: int):

    def db_operation():
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=dict_row)

        try:
            cursor.execute(
                """
                SELECT id, user_id, playlist_name, playlist_items
                FROM user_playlist
                WHERE id = %s
                ORDER BY created_at DESC
                """,
                (id,),
            )

            playlist = cursor.fetchone()
            if not playlist:
                raise HTTPException(status_code=404, detail="Playlist not found")

            selected_playlist = {
                "id": playlist["id"],
                "user_id": playlist["user_id"],
                "playlist_name": playlist["playlist_name"],
                "playlist_items": playlist["playlist_items"],
            }

            items = get_song_playlist_items_by_id(conn, playlist["id"])

            return {"playlist": selected_playlist, "items": items}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            cursor.close()
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)
