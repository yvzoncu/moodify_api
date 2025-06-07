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
from emotion_detector import search_songs_with_embedding, worker
from psycopg.rows import dict_row
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from uuid import uuid4
from datetime import datetime, UTC


class PlaylistItem(BaseModel):
    song_id: int


class CreatePlaylistRequest(BaseModel):
    user_id: str
    playlist_name: str
    playlist_items: List[PlaylistItem]


class SharePlaylistRequest(BaseModel):
    playlist_id: int
    user_id: str
    user_name: str


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
async def search(query: str, user_id: str = "1", k: int = 5):
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
                JOIN song_data s ON (item->>'song_id')::INT = s.id
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
                        "song_info": song["song_info"],
                        "tempo": song["tempo"],
                        "danceability": song["danceability"],
                        "energy": song["energy"],
                        "acousticness": song["acousticness"],
                        "valence": song["valence"],
                        "release_year": song["release_year"],
                        "genre": song["genre"],
                        "album_image": song["album_image"],
                        "spotify_id": song["spotify_id"],
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
                    SELECT 
                    p.id AS playlist_id,
                    p.user_id,
                    p.playlist_name,
                      (
                        SELECT json_agg(json_build_object(
                          'id', s.id,
                          'song', s.song,
                          'artist', s.artist,
                          'genre', s.genre,
                          'tempo', s.tempo,
                          'danceability', s.danceability,
                          'energy', s.energy,
                          'valence', s.valence,
                          'acousticness', s.acousticness,
                          'release_year', s.release_year,
                          'album_image', s.album_image,
                          'spotify_id', s.spotify_id
                        ))
                        FROM jsonb_array_elements(p.playlist_items) AS item
                        JOIN song_data s ON (item->>'song_id')::INT = s.id
                      ) AS songs
                    FROM user_playlist p
                    WHERE p.user_id = %s
                    ORDER BY p.created_at DESC;
                    """,
                    (user_id,),
                )

                playlists = []
                for row in cursor.fetchall():
                    playlists.append(
                        {
                            "id": row["playlist_id"],
                            "user_id": row["user_id"],
                            "playlist_name": row["playlist_name"],
                            "playlist_items": row["songs"],
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
                    "song_info": song["song_info"],
                    "tempo": song["tempo"],
                    "danceability": song["danceability"],
                    "energy": song["energy"],
                    "acousticness": song["acousticness"],
                    "valence": song["valence"],
                    "release_year": song["release_year"],
                }

                return song_item

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-playlist-by-playlist-id")
async def get_playlist_by_id(id: int):
    """
    Get playlist and its songs by playlist ID
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
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/web-search")
async def web_search(query: str):
    """
    Search for songs on the web and add them to the database
    """
    loop = asyncio.get_event_loop()

    def search_operation():
        try:
            return worker(query)
        except Exception as e:
            print(f"Error in web search: {e}")
            return None

    await loop.run_in_executor(executor, search_operation)

    # After adding new songs, search in our database
    songs = await search(query=query, user_id="system", k=5)
    return songs


@app.patch("/api/update-song-spotify-info")
async def update_song_spotify_info(id: int):
    """
    Update spotify_id and album_image for a song in song_data by id using Spotify API.
    If both values are already set, return them immediately.
    Always return only spotify_id and album_image in the response.
    """

    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # 1. Fetch song, artist, spotify_id, album_image
                cursor.execute(
                    "SELECT id, song, artist, spotify_id, album_image FROM song_data WHERE id = %s",
                    (id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Song not found.")
                song, artist = row["song"], row["artist"]
                spotify_id, album_image = row["spotify_id"], row["album_image"]

                # 2. If both are set, return immediately
                if spotify_id and album_image:
                    return {"spotify_id": spotify_id, "album_image": album_image}

                # 3. Setup Spotify client
                client_id = os.getenv("SPOTIFY_CLIENT_ID")
                client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
                if not client_id or not client_secret:
                    raise HTTPException(
                        status_code=500, detail="Spotify credentials not set in .env"
                    )
                sp = Spotify(
                    auth_manager=SpotifyClientCredentials(client_id, client_secret)
                )

                # 4. Search Spotify
                query = f"track:{song} artist:{artist}"
                results = sp.search(q=query, type="track", limit=1)
                items = results.get("tracks", {}).get("items", [])
                if not items:
                    raise HTTPException(
                        status_code=404, detail="No Spotify match found."
                    )
                track = items[0]
                new_spotify_id = track["id"]
                images = track["album"]["images"]
                new_album_image = (
                    images[-1]["url"] if images else None
                )  # Smallest image

                # 5. Update DB
                cursor.execute(
                    "UPDATE song_data SET spotify_id = %s, album_image = %s WHERE id = %s RETURNING spotify_id, album_image",
                    (new_spotify_id, new_album_image, id),
                )
                updated = cursor.fetchone()
                conn.commit()
                return {
                    "spotify_id": updated["spotify_id"],
                    "album_image": updated["album_image"],
                }
        except HTTPException:
            raise
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()

    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.post("/api/share-playlist")
async def share_playlist(request: SharePlaylistRequest):
    """
    Generate a share token for a playlist and return the shareable URL.
    """
    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Check if a share token already exists for this playlist and user
                cursor.execute(
                    """
                    SELECT share_token FROM share_playlist WHERE playlist_id = %s AND owner_user_id = %s
                    """,
                    (request.playlist_id, request.user_id),
                )
                row = cursor.fetchone()
                if row:
                    share_token = row["share_token"]
                    share_url = f"http://192.168.10.150:3000/share/{share_token}"
                    return {"share_token": share_token, "share_url": share_url, "user_name": request.user_name}
                # Generate unique token and create new record
                share_token = str(uuid4())
                created_at = datetime.now(UTC)
                cursor.execute(
                    """
                    INSERT INTO share_playlist (share_token, playlist_id, owner_user_id, user_name, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING share_token
                    """,
                    (share_token, request.playlist_id, request.user_id, request.user_name, created_at),
                )
                conn.commit()
                share_url = f"http://192.168.10.150:3000/share/{share_token}"
                return {"share_token": share_token, "share_url": share_url, "user_name": request.user_name}
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)


@app.get("/api/get-shared-playlist")
async def get_shared_playlist(share_token: str):
    """
    Get shared playlist details by share token (UUID).
    """
    def db_operation():
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT playlist_id, owner_user_id FROM share_playlist WHERE share_token = %s
                    """,
                    (share_token,),
                )
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Share token not found")
                playlist_id = row["playlist_id"]
                owner_user_id = row["owner_user_id"]
                # Fetch playlist details
                cursor.execute(
                    """
                    SELECT id, user_id, playlist_name, playlist_items, user_name
                    FROM user_playlist
                    WHERE id = %s
                    """,
                    (playlist_id,),
                )
                playlist = cursor.fetchone()
                if not playlist:
                    raise HTTPException(status_code=404, detail="Playlist not found")
                items = get_song_playlist_items_by_id(conn, playlist_id)
                return {
                    "playlist": {
                        "id": playlist["id"],
                        "user_id": playlist["user_id"],
                        "playlist_name": playlist["playlist_name"],
                        "user_name": playlist["user_name"],
                    },
                    "items": items,
                    "owner_user_id": owner_user_id,
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
    return await asyncio.get_event_loop().run_in_executor(executor, db_operation)
