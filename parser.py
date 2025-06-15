from playwright.sync_api import sync_playwright
import json
import psycopg
from dotenv import load_dotenv
import os
from psycopg.rows import dict_row
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
import requests
from pgvector.psycopg import register_vector
import time

load_dotenv()


DB_HOST = os.getenv("DB_HOST", "moodify.cje0wa8qioij.eu-north-1.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "lyricsdb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    exit(1)

client = OpenAI(api_key=API_KEY)


@dataclass
class SongData:
    song: str
    artist: str
    song_info: Optional[str]
    genre: Optional[str]
    tempo: Optional[str]
    danceability: Optional[str]
    energy: Optional[str]
    acousticness: Optional[str]
    valence: Optional[str]
    release_year: Optional[str]
    embeddings: Optional[str]
    album_image: Optional[str]
    spotify_id: Optional[str]
    music_key: Optional[str]
    camelot_value: Optional[str]
    album: Optional[str]
    tempo_and_key_analysis: Optional[str]


def get_db_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        row_factory=dict_row,
    )
    

def extract_track_urls_from_page(url: str) -> list:

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Wait for the elements to load
        page.wait_for_selector("td.table_img a", timeout=10000)

        # Query all anchor tags within the specific td class
        track_elements = page.query_selector_all("td.table_img a")
        track_urls = [element.get_attribute("href") for element in track_elements if element.get_attribute("href")]

        browser.close()
        return track_urls
    


def scrape_songdata(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                # Navigate to the page with timeout
                page.goto(url, timeout=30000)
                
                # Ensure content has loaded with error handling
                try:
                    page.wait_for_selector("div#review_section > div p", timeout=10000)
                except Exception as e:
                    print(f"Warning: Review section not found for {url}: {e}")
                
                try:
                    page.wait_for_selector("dl.grid dt", timeout=10000)
                except Exception as e:
                    print(f"Warning: Grid data not found for {url}: {e}")
                    
                try:
                    page.wait_for_selector("script[type='application/ld+json']", state="attached", timeout=10000)
                except Exception as e:
                    print(f"Warning: JSON-LD script not found for {url}: {e}")

                # 1. First two <p> tags in the nested div
                paragraphs = []
                try:
                    p_tags = page.query_selector_all("div#review_section > div p")[:2]
                    paragraphs = [p.inner_text().strip() for p in p_tags]
                except Exception as e:
                    print(f"Warning: Could not extract paragraphs for {url}: {e}")

                # 2. Scrape all dt & dd under dl.grid
                items = []
                try:
                    dts = page.query_selector_all("dl.grid dt")
                    for dt in dts:
                        try:
                            dd = dt.evaluate_handle("el => el.nextElementSibling")
                            if dd:
                                items.append({
                                    "label": dt.inner_text().strip(),
                                    "value": dd.inner_text().strip()
                                })
                        except Exception as e:
                            print(f"Warning: Could not extract grid item for {url}: {e}")
                            continue
                except Exception as e:
                    print(f"Warning: Could not extract grid data for {url}: {e}")
                
                # 3. Extract JSON-LD data
                jsonld_data = {}
                try:
                    jsonld_text = page.locator("script[type='application/ld+json']").nth(0).text_content()
                    jsonld_data = json.loads(jsonld_text)
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON-LD data for {url}: {e}")
                except Exception as e:
                    print(f"Warning: Could not extract JSON-LD for {url}: {e}")

                browser.close()
                return {
                    "first_two_paragraphs": paragraphs,
                    "dl_grid": items,
                    "jsonld_data": jsonld_data
                }
                
            except Exception as e:
                print(f"Error during page scraping for {url}: {e}")
                browser.close()
                return None
                
    except Exception as e:
        print(f"Error launching browser or connecting to {url}: {e}")
        return None
        
def song_info_searcher(song):
    try:
        search_response = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a music information researcher. Search the web for contextual information about songs. "
                        "Focus on reliable sources like AllMusic, Discogs, Wikipedia, Genius, official artist websites, and music databases. "
                        "Write a comprehensive but concise description that combines all the key information about the song.\n\n"
                        "Include the following elements in your response:\n"
                        "- Background story, creation details, or interesting facts\n"
                        "- Brief summary of lyrics themes and meaning\n"
                        "- Emotional context, tone, and mood\n"
                        "- Any notable achievements, chart performance, or cultural impact\n\n"
                        "Writing Guidelines:\n"
                        "1. Write in a flowing, informative style suitable for music app users\n"
                        "2. Keep it engaging but professional (5-15 sentences total)\n"
                        "3. Combine all information into one cohesive paragraph\n"
                        "4. Focus on what makes this song interesting or unique\n"
                        "5. Use present tense when describing the song's characteristics\n"
                        "6. Avoid technical jargon - write for general music listeners\n\n"
                        "Return ONLY the clean text description, no JSON, no formatting, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Please search for detailed information about the song: '{song}' and write a comprehensive description.\n\n"
                              f"Create a single, well-written paragraph that covers:\n"
                              f"- The story behind the song and interesting facts\n"
                              f"- What the lyrics are about and main themes\n"
                              f"- The emotional tone and mood of the song\n"
                              f"- Any notable achievements or cultural significance\n\n"
                              f"Write this as clean, engaging text that can be displayed directly to users in a music app."
                },
            ],
            web_search_options={},  # Enable search if needed
        )
        search_text = search_response.choices[0].message.content
        return search_text
    except Exception as e:
        print(f"API call failed: {e}")
        return None
    


def generate_tempo_and_key_analysis(song_data: SongData):
    """
    Use Mistral AI API to generate a tempo and music key analysis for a song.
    """

    prompt = f"""
Generate a detailed musical tempo and key analysis like a musicologist for the following song.

Song: {song_data.song}
Artist: {song_data.artist}
Tempo (BPM): {song_data.tempo}
Key: {song_data.music_key}
Camelot Key: {song_data.camelot_value if song_data.camelot_value else "N/A"}


Include in the output:
1. Tempo analysis including tempo marking (e.g., Allegro, Vivace), half-time, double-time, and if it's fast/slow.
2. Musical key analysis, including Camelot wheel interpretation, energy boost/drop transitions, and musical feel.
Output must be in two sections: "BPM and Tempo" and "Music Key".
Make it clear, rich, and natural-language like the following style:

---
BPM and Tempo
Since [Song] by [Artist] has a tempo of [X] beats per minute, the tempo markings of this song would be [Tempo Marking]...
---
"""

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-small",  # or mistral-medium, depending on access
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        }
    )

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Failed to generate analysis: {response.status_code} - {response.text}")
    
    

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return response.data[0].embedding


def build_embedding_string(song_data: SongData) -> str:
    parts = []

    # Required main info
    if song_data.song and song_data.artist:
        parts.append(f'"{song_data.song}" by {song_data.artist}')

    # Genre
    if song_data.genre:
        parts.append(f"Genre: {song_data.genre}")

    
    def get_descriptive_level(value, feature_type="general"):
        if value is None:
            return None
            
        # Convert string percentages to float if needed
        if isinstance(value, str):
            try:
                value = float(value) / 100.0  # Convert percentage string to decimal
            except ValueError:
                return None
        
        if not isinstance(value, (float, int)):
            return None
            
        # Clamp value to 0-1 range
        value = max(0, min(1, value))
        
        # Different scaling for tempo (BPM range)
        if feature_type == "tempo":
            # For tempo, value should be the actual BPM number
            try:
                bpm = float(song_data.tempo) if song_data.tempo else 0
                if bpm < 80:
                    return "very slow"
                elif bpm < 100:
                    return "slow"
                elif bpm < 120:
                    return "moderate"
                elif bpm < 140:
                    return "fast"
                else:
                    return "very fast"
            except:
                return "unknown tempo"
            
        if feature_type == "valence":
            if value < 0.2:
                return "very sad/negative"
            elif value < 0.4:
                return "sad/melancholic"
            elif value < 0.6:
                return "neutral mood"
            elif value < 0.8:
                return "happy/positive"
            else:
                return "very happy/euphoric"
                
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

    # Tempo
    if song_data.tempo:
        tempo_desc = get_descriptive_level(song_data.tempo, "tempo")
        parts.append(f"Tempo: {tempo_desc} ({song_data.tempo} BPM)")

    # Danceability
    if song_data.danceability:
        dance_desc = get_descriptive_level(song_data.danceability, "danceability")
        parts.append(f"Danceability: {dance_desc}")

    # Energy
    if song_data.energy:
        energy_desc = get_descriptive_level(song_data.energy, "energy")
        parts.append(f"Energy: {energy_desc}")

    # Acousticness
    if song_data.acousticness:
        acoustic_desc = get_descriptive_level(song_data.acousticness, "acousticness")
        parts.append(f"Acousticness: {acoustic_desc}")
        
    # Valence
    if song_data.valence:
        valence_desc = get_descriptive_level(song_data.valence, "valence")
        parts.append(f"Mood: {valence_desc}")

    # Release year
    if song_data.release_year:
        parts.append(f"Released in {song_data.release_year}")

    # Music key
    if song_data.music_key:
        parts.append(f"Key: {song_data.music_key}")

    # Album
    if song_data.album:
        parts.append(f"Album: {song_data.album}")

    # Song info - add if available
    if song_data.song_info:
        parts.append(f"Info: {song_data.song_info}")

    # Tempo and key analysis
    if song_data.tempo_and_key_analysis:
        parts.append(f"Tempo and Key Analysis: {song_data.tempo_and_key_analysis}")
        

    # Join with period + space for better clarity
    return ". ".join(parts)


def check_song_exists(song_data: SongData) -> bool:
    """Check if song already exists in database"""
    conn = get_db_connection()
    
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id FROM song_data WHERE LOWER(song) = LOWER(%s) AND LOWER(artist) = LOWER(%s)
            LIMIT 1
            """,
            (song_data.song, song_data.artist)
        )
        existing_song = cursor.fetchone()
        
    conn.close()
    return existing_song is not None


def write_song_to_db(song_data: SongData) -> bool:
    """Write song data to database"""
    conn = get_db_connection()
    register_vector(conn) 
    
    text = build_embedding_string(song_data)
    embedding = get_embedding(text)
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO song_data (song, artist, genre, song_info, album_image, release_year, energy, danceability, valence, tempo, music_key, camelot_value, acousticness, album, tempo_and_key_analysis, embeddings)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (song_data.song, song_data.artist, song_data.genre, song_data.song_info, song_data.album_image, song_data.release_year, 
                 song_data.energy, song_data.danceability, song_data.valence, song_data.tempo, 
                 song_data.music_key, song_data.camelot_value, song_data.acousticness, song_data.album, 
                 song_data.tempo_and_key_analysis, embedding)
            )
            conn.commit()
            print("Song saved to database")
            return True
    except Exception as e:
        print(f"Error saving song to database: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()
        
        




if __name__ == "__main__":
    
    url = "https://songdata.io/charts/norway"
    
    tracks = extract_track_urls_from_page(url)
    for index, track in enumerate(tracks):
        #wait 2 seconds
        time.sleep(2)
        base_url = "https://songdata.io"
        url = base_url + track
        data = scrape_songdata(url)
        if data is None:
            continue
      
        song_name = data["jsonld_data"]["name"]
        artist_name = data["jsonld_data"]["byArtist"]
        image_url = data["jsonld_data"]["image"]
        date_published = data["jsonld_data"]["datePublished"]

        energy_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'Energy'), None)
        if energy_value:
            energy_value = str(float(energy_value.rstrip('%')) / 100)

        danceability_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'Danceability'), None)
        if danceability_value:
            danceability_value = str(float(danceability_value.rstrip('%')) / 100)

        valence_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'Valence'), None)
        if valence_value:
            valence_value = str(float(valence_value.rstrip('%')) / 100)

        acousticness_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'Acousticness'), None)
        if acousticness_value:
            acousticness_value = str(float(acousticness_value.rstrip('%')) / 100)

        album_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'inAlbum'), None)
        camelot_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'Camelot'), None)
        musical_key_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'Key'), None)
        tempo_value = next((item['value'] for item in data["dl_grid"] if item['label'] == 'BPM'), None)



        # Create SongData instance
        song_data = SongData(
            song=song_name,
            artist=artist_name,
            song_info=None, 
            genre=None,  # Not extracted yet
            tempo=tempo_value,
            danceability=danceability_value,
            energy=energy_value,
            acousticness=acousticness_value,
            valence=valence_value,
            release_year=date_published,
            embeddings=None,  # Not extracted yet
            album_image=image_url,
            spotify_id=None,  # Not extracted yet
            music_key=musical_key_value,
            camelot_value=camelot_value,
            album=album_value,
            tempo_and_key_analysis=None
        )

        if check_song_exists(song_data):
            print("Song already exists")
        else:
            song_data.song_info = song_info_searcher(f"{song_name} by {artist_name}")
            song_data.tempo_and_key_analysis = generate_tempo_and_key_analysis(song_data)
            write_song_to_db(song_data)
            print(f"Song {index + 1} saved to database")
            
    
    

    
    

    
    
