import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
import time

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
    "description": "Extract AUDIO FEATURES and OTHER SONG INFO for each song",
    "parameters": {
      "type": "object",
      "properties": {
        "songs": {
          "type": "array",
          "description": "A list of structured song metadata and audio analysis.",
          "items": {
            "type": "object",
            "properties": {
              "song": {
                "type": "string",
                "description": "Title of the song."
              },
              "artist": {
                "type": "string",
                "description": "The performing artist of the song."
              },
              "genre": {
                "type": "string",
                "description": "Primary genre (e.g., Pop, Rock, Hip-Hop, Country)."
              },
              "release_year": {
                "type": "string",
                "description": "Release year of the song or album."
              },
              "tempo": {
                "type": "number",
                "description": "Beats per minute (BPM) of the song."
              },
              "danceability": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "How suitable the song is for dancing, from 0.0 (least) to 1.0 (most)."
              },
              "energy": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Perceived intensity and activity level, from 0.0 to 1.0."
              },
              "acousticness": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence measure of whether the track is acoustic."
              },
              "valence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Valence (mood) score (0.0 to 1.0)."
              },
               "music_key": {
                "type": "string",
                "description": "Musical key of the song (e.g., C Major, A minor)."
              },
              "other_song_info": {
                "type": "string",
                "description": "All information under OTHER SONG INFO section."
              }
             
            },
            "required": ["song", "artist", "genre"]
          }
        }
      },
      "required": ["songs"]
    }
  }
}

    
       
]


def song_suggester(user_prompt):
    try:
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
        search_response = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a comprehensive music data researcher and song attribute analyzer. Search the web (especially songdata.io). "
                        "For each song provided by the user, return **two structured sections**:\n\n"

                        "**1. AUDIO FEATURES (high priority)**:\n"
                        "- Genre: Primary genre (e.g., Pop, Rock, Hip-Hop, Country).\n"
                        "- Tempo (BPM) \n"
                        "- Danceability: Numeric value from 0.0 (least) to 1.0 (most).\n"
                        "- Energy: Numeric value from 0.0 to 1.0.\n"
                        "- Acousticness: Numeric value from 0.0 to 1.0.\n"
                        "- Valence: Numeric value from 0.0 to 1.0.\n"
                        "- Musical key: Musical key of the song (e.g., C Major, A minor).\n"
                        
                        "**2. OTHER SONG INFO (in-depth analysis)**:\n"
                        "- Song name\n"
                        "- Artist name\n"
                        "- Album name\n"
                        "- Release date\n"
                        "- Duration\n"
                        "- Track number and total tracks on the album\n"
                        "- Country of origin\n"
                        "- Lyrics summary\n"
                        "- Emotional context\n"
                        "- Popularity\n"
                        "- Funny facts (if available)\n"
                        "- Unique facts or reception (if available)\n\n"
                        
                        "Do not skip any detail if it’s available."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{song}",
                },
            ],
            web_search_options={},  # Enable search if needed
        )
        search_text = search_response.choices[0].message.content
        return search_text
    except Exception as e:
        print(f"API call failed: {e}")
        return None



def song_attribute_extractor(search_text, songs):
    try:

        songs_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (f"Extract AUDIO FEATURES and OTHER SONG INFO for each song from this search result. Return AUDIO FEATURES as seperate property and OTHER SONG INFO as one property. Now process following text: {search_text}."),
                },
                {"role": "user", "content": f"find attributes for following songs: {songs}"},
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


def build_embedding_string(song_data: dict) -> str:
    parts = []

    # Required main info
    if song_data.get("song") and song_data.get("artist"):
        parts.append(f'"{song_data["song"]}" by {song_data["artist"]}')

    # Genre
    if song_data.get("genre"):
        parts.append(f"Genre: {song_data['genre']}")

    
    def get_descriptive_level(value, feature_type="general"):
        if value is None or not isinstance(value, (float, int)):
            return None
            
        # Clamp value to 0-1 range
        value = max(0, min(1, value))
        
        # Different scaling for tempo (BPM range)
        if feature_type == "tempo":
            if value < 80:
                return "very slow"
            elif value < 100:
                return "slow"
            elif value < 120:
                return "moderate"
            elif value < 140:
                return "fast"
            else:
                return "very fast"
            
        if feature_type == "valence":
            if value < 0.2:
                valence_label = "very sad/negative"
            elif value < 0.4:
                valence_label = "sad/melancholic"
            elif value < 0.6:
                valence_label = "neutral mood"
            elif value < 0.8:
                valence_label = "happy/positive"
            else:
                valence_label = "very happy/euphoric"
                
        # Standard 0-1 scale for audio features
        if value < 0.2:
            return "very low"
        elif value < 0.4:
            return "low"
        elif value < 0.6:
            return "moderate"
        elif value < 0.8:
            return "high"
        else:
            return "very high"

    tempo = song_data.get("tempo")
    if tempo is not None:
        tempo_desc = get_descriptive_level(tempo, "tempo")
        parts.append(f"Tempo: {tempo_desc} ({tempo} BPM)")

    danceability = song_data.get("danceability")
    if danceability is not None:
        dance_desc = get_descriptive_level(danceability, "danceability")
        parts.append(f"Danceability: {dance_desc}")

    energy = song_data.get("energy")
    if energy is not None:
        energy_desc = get_descriptive_level(energy, "energy")
        parts.append(f"Energy: {energy_desc}")

    acousticness = song_data.get("acousticness")
    if acousticness is not None:
        acoustic_desc = get_descriptive_level(acousticness, "acousticness")
        parts.append(f"Acousticness: {acoustic_desc}")
        
    valence = song_data.get("valence")
    if valence is not None:
        valence_desc = get_descriptive_level(valence, "valence")
        parts.append(f"Mood: {valence_desc}")

    # Release year
    release_year = song_data.get("release_year")
    if release_year:
        parts.append(f"Released in {release_year}")

    # music_key
    music_key = song_data.get("music_key")
    if music_key:
        parts.append(f"Key: {music_key}")

    # song_info - truncate if too long
    song_info = song_data.get("song_info")
    
    parts.append(f"Info: {song_info}")

    # Join with period + space for better clarity
    return ". ".join(parts)


def insert_song_data(song_data):
    conn = get_db_connection()
    register_vector(conn) 

    text = build_embedding_string(song_data)
    embedding = get_embedding(text)
    print(text)

    try:
        with conn.cursor() as cursor:
            # Check if song already exists
            cursor.execute(
                """
                SELECT id, song, artist, song_info, genre, tempo,
                       danceability, energy, acousticness, valence,
                       release_year, music_key FROM song_data
                WHERE LOWER(song) = LOWER(%s) AND LOWER(artist) = LOWER(%s)
                LIMIT 1
                """,
                (song_data["song"], song_data["artist"]),
            )

            existing_song = cursor.fetchone()
            if existing_song:
                # Return existing song data
                return {
                    "song_id": existing_song["id"],
                    "song": existing_song["song"],
                    "artist": existing_song["artist"],
                    "genre": existing_song["genre"],
                    "tempo": existing_song["tempo"],
                    "danceability": existing_song["danceability"],
                    "energy": existing_song["energy"],
                    "acousticness": existing_song["acousticness"],
                    "valence": existing_song["valence"],
                    "song_info": existing_song["song_info"],
                    "release_year": existing_song["release_year"],
                    "music_key": existing_song["music_key"],
                    "distance": 0.0,
                }

            # Insert the song and embedding
            insert_query = sql.SQL(
                """
                INSERT INTO song_data (
                    song, artist, song_info, genre, tempo,
                    danceability, energy, acousticness, valence,
                    release_year, music_key, embeddings
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, song, artist, song_info, genre, tempo,
                         danceability, energy, acousticness, valence,
                         release_year, music_key
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
                    song_data.get("music_key"),
                    embedding,
                ),
            )
            
            # Get the inserted song data
            new_song = cursor.fetchone()
            conn.commit()
            
            # Return the newly inserted song data
            return {
                "song_id": new_song["id"],
                "song": new_song["song"],
                "artist": new_song["artist"],
                "genre": new_song["genre"],
                "tempo": new_song["tempo"],
                "danceability": new_song["danceability"],
                "energy": new_song["energy"],
                "acousticness": new_song["acousticness"],
                "valence": new_song["valence"],
                "song_info": new_song["song_info"],
                "release_year": new_song["release_year"],
                "music_key": new_song["music_key"],
                "distance": 0.0,
            }

    except Exception as e:
        conn.rollback()
        print("Insert failed:", e)
        return None
    finally:
        conn.close()


def search_songs_with_embedding(user_prompt, top_k, threshold=0.7):

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
            for song_item in song_list:
                cursor.execute(
                    """
                    SELECT * FROM song_data
                    WHERE LOWER(song) = LOWER(%s) AND LOWER(artist) = LOWER(%s)
                    LIMIT 1
                    """,
                    (song_item["song"], song_item["artist"]),
                )
                existing_song = cursor.fetchone()
                if existing_song:
                    existing_songs.append({
                            "song_id": existing_song["id"],
                            "song": existing_song["song"],
                            "artist": existing_song["artist"],
                            "genre": existing_song["genre"],
                            "tempo": existing_song["tempo"],
                            "danceability": existing_song["danceability"],
                            "energy": existing_song["energy"],
                            "acousticness": existing_song["acousticness"],
                            "valence": existing_song["valence"],
                            "song_info": existing_song["song_info"],
                            "release_year": existing_song.get("release_year"),
                            "music_key": existing_song.get("music_key"),
                            "distance": 0.0,
                        })
                else:
                    new_songs.append(song_item)
    except Exception as e:
        print(f"Error in split_existing_and_new_songs: {e}")
        return None
    finally:
        conn.close()

    return existing_songs, new_songs


def worker(user_prompt):
    """
    Main worker function that searches web for songs, extracts attributes, and adds to database
    Returns list of songs (existing + newly added) or None if error
    """
    print(f"Starting worker for prompt: {user_prompt}")
    
    get_suggestions = song_suggester(user_prompt)
    if not get_suggestions:
        print("No web suggestions found")
        return None
    print(get_suggestions)
    print("#"*100)
    
    song_info = song_attribute_finder(get_suggestions)
    if song_info is None:
            print("No song info found")
            return None
    print(song_info)
    print("#"*100)
    
    song_attributes = song_attribute_extractor(song_info, get_suggestions)
    if song_attributes is None:
        print("No song attributes extracted")
        return None
    print(song_attributes)
    print("#"*100)
    
    
   
    
    


def add_song_to_db(song_and_artist):
    """
    Add a single song to database by song and artist string
    """
    print(f"Adding song to DB: {song_and_artist}")
    song_info = song_attribute_finder(song_and_artist)
    if song_info is None:
        print("No song info found")
        return None
    
    print(song_info)
    print("#"*100)
    
    song_attributes = song_attribute_extractor(song_info)
    if song_attributes is None:
        print("No song attributes extracted")
        return None
    print(song_attributes)
    print("#"*100)
    
    attrs = song_attributes[0]
    if not all(key in attrs and attrs[key] is not None for key in ['genre', 'tempo', 'danceability', 'energy', 'acousticness', 'valence']):
        print(f"Missing essential attributes for {attrs['song']} by {attrs['artist']}")
        return None
    
    result = insert_song_data(attrs)
    if result:
        print(f"Successfully added: {result['song']} by {result['artist']}")
        return result
    else:
        print("Failed to insert song")
        return None


def user_intent_extractor(search_text):
    try:

        user_intent = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                 {
            "role": "system",
            "content": """
You are a music search assistant.

Your job is to extract structured information from a user's prompt for searching songs in a database. Your response must return:

- A list of songs (if mentioned)
- A list of artists (if mentioned)
- Always return estimated numeric values for: danceability, energy, acousticness, and valence — even if they must be inferred from vague language like mood, adjectives, or genre.

ONLY include a "Custom-request" value if the user is asking for something non-musical or factual, such as:
- "Songs that won Eurovision"
- "Top 2 songs on Spotify this week"
- "Most streamed tracks of all time"

If the user prompt is:
- Non-moral (e.g., offensive, hateful, or unethical): return empty arrays and 0.0 values with an empty Custom-request.
- Not meaningful (e.g., gibberish or unrelated to music): return empty arrays and 0.0 values with an empty Custom-request.

Never make up songs, artists, or rankings. If uncertain, leave arrays empty and return best-effort audio feature estimates.
"""
        },
                {"role": "user", "content": search_text},
            ],
            tools=functions,
            tool_choice={"type": "function", "function": {"name": "analyze_user_intent"}},
        )

        # Check if tool_calls exists
        if not user_intent.choices[0].message.tool_calls:

            return None

        # Extract the function call arguments (JSON string)
        tool_call = user_intent.choices[0].message.tool_calls[0]

        try:
            data = json.loads(tool_call.function.arguments)
            
        except json.JSONDecodeError as e:
            return None

        return data

    except Exception as e:
        print(f"API call failed: {e}")
        return None
    



 