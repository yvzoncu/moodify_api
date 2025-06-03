import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector


load_dotenv()


DB_HOST = os.getenv("DB_HOST", "moodify.cje0wa8qioij.eu-north-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "lyricsdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_db_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        row_factory=dict_row,
    )


# Check both possible environment variable names
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")

if not API_KEY:
    exit(1)

client = OpenAI(api_key=API_KEY)

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_song_titles",
            "description": "Extract a list of real, recent songs and their artists based on the user's request or search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "songs": {
                        "type": "array",
                        "description": "List of song and artist pairs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "song": {
                                    "type": "string",
                                    "description": "The title of the song.",
                                },
                                "artist": {
                                    "type": "string",
                                    "description": "The artist who performed the song.",
                                },
                            },
                            "required": ["song", "artist"],
                        },
                    }
                },
                "required": ["songs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "song_analyser",
            "description": "Extract structured audio and metadata attributes from a list of real songs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "songs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "song": {
                                    "type": "string",
                                    "description": "Title of the song.",
                                },
                                "artist": {
                                    "type": "string",
                                    "description": "The performing artist of the song.",
                                },
                                "song_info": {
                                    "type": "string",
                                    "description": "All useful information about the song use as much info as possible from input. (e.g., chart position, album, popularity).",
                                },
                                "genre": {
                                    "type": "string",
                                    "description": "Primary genre (e.g., Pop, Rock, Hip-Hop, Country).",
                                },
                                "release_year": {
                                    "type": "string",
                                    "description": "Release yar of the song or album.",
                                },
                                "tempo": {
                                    "type": "number",
                                    "description": "Beats per minute (BPM) of the song.",
                                },
                                "danceability": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "How suitable the song is for dancing, from 0.0 (least) to 1.0 (most).",
                                },
                                "energy": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "Perceived intensity and activity level, from 0.0 to 1.0.",
                                },
                                "acousticness": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "Confidence measure of whether the track is acoustic.",
                                },
                                "valence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "Valence (mood) score (0.0 to 1.0).",
                                },
                            },
                            "required": [
                                "song",
                                "artist",
                                "genre",
                            ],
                        },
                        "description": "A list of structured song metadata and audio analysis.",
                    }
                },
                "required": ["songs"],
            },
        },
    },
]


def song_suggester(user_prompt):
    try:
        print("\U0001f310 Step 1: Searching the web for real songs...\n")
        search_response = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a music expert assistant. Use reliable sources like Spotify and Popnable "
                        "to find up to 3 (if not requested otherwise) recent, real songs that closely match the user's request."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Find up to 3 (if not requested otherwise) recent songs with title and artist matching: {user_prompt}",
                },
            ],
            web_search_options={},
        )
        search_text = search_response.choices[0].message.content
        print("\U0001f50d Found via web search:\n", search_text, "\n")
        return search_text
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def song_and_artist_extractor(search_text):
    try:

        songs_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": ("Extract song attributes from this search result."),
                },
                {"role": "user", "content": search_text},
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "get_song_titles"}},
        )

        # Check if tool_calls exists
        if not songs_response.choices[0].message.tool_calls:

            return None

        # Extract the function call arguments (JSON string)
        tool_call = songs_response.choices[0].message.tool_calls[0]

        try:
            data = json.loads(tool_call.function.arguments)
            songs = data.get("songs", [])
        except json.JSONDecodeError as e:
            return None

        return songs

    except Exception as e:
        print(f"API call failed: {e}")
        return None


def song_attribute_finder(song):
    try:
        print("\U0001f310 Step 1: Searching the web for real songs...\n")
        search_response = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an attribute finder. Search the web (especially songdata.io) for audio features "
                        "of the given song. Include attributes like tempo, BPM, key, energy, valence, danceability, etc."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Find detailed audio features of the song: {song}",
                },
            ],
            web_search_options={},
        )
        search_text = search_response.choices[0].message.content
        return search_text
    except Exception as e:
        print(f"API call failed: {e}")
        return None


def song_attribute_extractor(search_text):
    try:

        songs_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": ("Extract song attributes from this search result."),
                },
                {"role": "user", "content": search_text},
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "song_analyser"}},
        )

        # Check if tool_calls exists
        if not songs_response.choices[0].message.tool_calls:

            return None

        # Extract the function call arguments (JSON string)
        tool_call = songs_response.choices[0].message.tool_calls[0]

        try:
            data = json.loads(tool_call.function.arguments)
            songs = data.get("songs", [])
        except json.JSONDecodeError as e:
            print(e)
            return None

        return songs

    except Exception as e:
        print(f"API call failed: {e}")
        return None


def streaming_song_analysis(user_prompt, songs):
    song_descriptions = []

    for s in songs:
        parts = [f"'{s['song']}' by {s['artist']}"]

        # Add attributes only if they exist and are not None
        optional_attributes = {
            "Genre": s.get("genre"),
            "Tempo": s.get("tempo"),
            "Danceability": s.get("danceability"),
            "Energy": s.get("energy"),
            "Acousticness": s.get("acousticness"),
            "Valence": s.get("valence"),
            "Info": s.get("song_info"),
        }

    attr_parts = [f"{k}: {v}" for k, v in optional_attributes.items() if v is not None]
    if attr_parts:
        parts.append(f"({', '.join(attr_parts)})")

    song_descriptions.append(" ".join(parts))

    try:
        explanation_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a passionate music expert. Explain in a natural, human tone (max 100 words) "
                        "why these songs perfectly match the user's request. Be friendly and enthusiastic."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Why are these songs ({song_descriptions}) a great match for: {user_prompt}?",
                },
            ],
            stream=True,
        )

        explanation = ""
        for chunk in explanation_response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                explanation += content
        return songs, explanation

    except Exception as e:
        return None


def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return response.data[0].embedding


def build_embedding_string(song_data: dict, max_info_length=200) -> str:
    parts = []

    # Required main info
    if song_data.get("song") and song_data.get("artist"):
        parts.append(f'"{song_data["song"]}" by {song_data["artist"]}')

    # Genre
    if song_data.get("genre"):
        parts.append(f"Genre: {song_data['genre']}")

    # Numeric audio features - round to 2 decimals
    def format_num(field):
        val = song_data.get(field)
        return f"{val:.2f}" if isinstance(val, (float, int)) else None

    tempo = song_data.get("tempo")
    if tempo is not None:
        parts.append(f"Tempo: {tempo} BPM")

    danceability = format_num("danceability")
    if danceability:
        parts.append(f"Danceability: {danceability}")

    energy = format_num("energy")
    if energy:
        parts.append(f"Energy: {energy}")

    acousticness = format_num("acousticness")
    if acousticness:
        parts.append(f"Acousticness: {acousticness}")

    valence = format_num("valence")
    if valence:
        parts.append(f"Valence: {valence}")

    # Release year
    release_year = song_data.get("release_year")
    if release_year:
        parts.append(f"Released in {release_year}")

    # song_info - truncate if too long
    song_info = song_data.get("song_info")
    if song_info:
        truncated_info = (
            (song_info[:max_info_length] + "...")
            if len(song_info) > max_info_length
            else song_info
        )
        parts.append(f"Info: {truncated_info}")

    # Join with period + space for better clarity
    return ". ".join(parts)


def insert_song_data(song_data):
    conn = get_db_connection()
    register_vector(conn)  # ✅ Register pgvector support

    text = build_embedding_string(song_data)
    embedding = get_embedding(text)

    try:
        with conn.cursor() as cursor:
            # Check if song already exists
            cursor.execute(
                """
                SELECT 1 FROM song_data
                WHERE LOWER(song) = LOWER(%s) AND LOWER(artist) = LOWER(%s)
                LIMIT 1
                """,
                (song_data["song"], song_data["artist"]),
            )

            if cursor.fetchone():
                return  # Skip insert

            # Insert the song and embedding
            insert_query = sql.SQL(
                """
                INSERT INTO song_data (
                    song, artist, song_info, genre, tempo,
                    danceability, energy, acousticness, valence,
                    release_year, embeddings
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            )

            cursor.execute(
                insert_query,
                (
                    song_data.get("song"),
                    song_data.get("artist"),
                    song_data.get("song_info"),
                    song_data.get("genre"),
                    song_data.get("tempo"),
                    song_data.get("danceability"),
                    song_data.get("energy"),
                    song_data.get("acousticness"),
                    song_data.get("valence"),
                    song_data.get("release_year"),
                    embedding,
                ),
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        print("Insert failed:", e)
    finally:
        conn.close()


def search_songs_with_embedding(user_prompt, top_k, threshold=0.7):
    print(user_prompt)
    user_embedding = get_embedding(user_prompt)

    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            # Using cosine distance operator <=> instead of negative inner product <#>
            cursor.execute(
                """
                SELECT id, song, artist, song_info, genre, tempo, 
                       danceability, energy, acousticness, valence, 
                       embeddings <=> %s::vector AS distance   
                FROM song_data
                WHERE embeddings <=> %s::vector <= %s
                ORDER BY distance
                LIMIT %s;
                """,
                (user_embedding, user_embedding, threshold, top_k),
            )

            result = cursor.fetchall()
            print(result)

            songs = []
            if result:
                for song in result:
                    songs.append(
                        {
                            "song_id": song["id"],
                            "song": song["song"],
                            "artist": song["artist"],
                            "genre": song["genre"],
                            "tempo": song["tempo"],
                            "danceability": song["danceability"],
                            "energy": song["energy"],
                            "acousticness": song["acousticness"],
                            "valence": song["valence"],
                            "song_info": song["song_info"],
                            "distance": song["distance"],
                        }
                    )
            else:
                print("no items found")

            return songs
    except Exception as e:
        print(e)
        conn.rollback()
        return None
    finally:
        conn.close()


def split_existing_and_new_songs(song_list):
    """
    Checks which songs from the list already exist in the database.

    Returns:
        A tuple of two lists:
        - existing_songs: songs found in the database
        - new_songs: songs not found in the database
    """
    conn = get_db_connection()
    existing_songs = []
    new_songs = []

    try:
        with conn.cursor() as cursor:
            for song in song_list:
                cursor.execute(
                    """
                    SELECT * FROM song_data
                    WHERE LOWER(song) = LOWER(%s) AND LOWER(artist) = LOWER(%s)
                    LIMIT 1
                    """,
                    (song["song"], song["artist"]),
                )
                result = cursor.fetchone()
                if result:
                    existing_songs.append(dict(result))
                else:
                    new_songs.append(song)
    except Exception as e:
        return None
    finally:
        conn.close()

    return existing_songs, new_songs


def worker(user_prompt):
    get_suggestions = song_suggester(user_prompt)
    print("Suggested songs:", get_suggestions)
    if not get_suggestions:
        return None
    suggested_songs = song_and_artist_extractor(get_suggestions)
    print("Suggested songs dettails:", suggested_songs)
    if not suggested_songs:
        return None
    existing_songs, new_songs = split_existing_and_new_songs(suggested_songs)
    if not new_songs:
        return None
    for song in new_songs:
        song_info = song_attribute_finder(f"{song['song']} by {song['artist']}")
        song_attributes = song_attribute_extractor(song_info)
        insert_song_data(song_attributes[0])
    total_songs = new_songs + existing_songs
    return total_songs


def add_song_to_db(song_and_artist):
    song_info = song_attribute_finder(song_and_artist)
    if song_info is None:
        return
    print("Song info:", song_info)
    song_attributes = song_attribute_extractor(song_info)
    if song_attributes is None:
        print("problem")
        return
    print("Song attributes:", song_attributes)
    insert_song_data(song_attributes[0])
