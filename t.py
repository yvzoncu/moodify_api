import os
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define function schemas
functions = [
    {
        "name": "get_song_by_name",
        "description": "Find a song by its name.",
        "parameters": {
            "type": "object",
            "properties": {
                "song_name": {"type": "string", "description": "The name of the song"},
            },
            "required": ["song_name"],
        },
    },
    {
        "name": "get_songs_by_artist",
        "description": "Find songs by a specific artist.",
        "parameters": {
            "type": "object",
            "properties": {
                "artist_name": {"type": "string", "description": "The artist's name"},
            },
            "required": ["artist_name"],
        },
    },
    {
        "name": "query_by_attributes",
        "description": "Query songs based on musical attributes",
        "parameters": {
            "type": "object",
            "properties": {
                "tempo": {"type": "string", "enum": ["slow", "medium", "fast"]},
                "energy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "danceability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "valence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["tempo"],
        },
    },
]


# Mock backend implementations
def get_song_by_name(song_name):
    return [f"🎵 Found song: '{song_name}'"]


def get_songs_by_artist(artist_name):
    return [f"🎤 Top songs by {artist_name}: [Song A, Song B, Song C]"]


def query_by_attributes(tempo, energy=None, danceability=None, valence=None):
    return [
        f"🔍 Found songs with tempo={tempo}, energy={energy}, danceability={danceability}, valence={valence}"
    ]


def handle_user_prompt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that interprets user requests for music. "
                    "Based on the user's description, infer the user's intent and extract musical attributes "
                    "such as tempo (slow, medium, fast), energy (0.0 to 1.0), danceability (0.0 to 1.0), "
                    "and valence (0.0 to 1.0). Provide the full set of attributes in your function call, "
                    "even if some must be estimated. For example, if the user says 'upbeat dance song with high energy', "
                    "tempo would be 'fast', energy might be 0.8 or higher, danceability high, valence high."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        functions=functions,
        function_call="auto",
    )

    message = response.choices[0].message

    # Check if model wants to call a function
    if message.function_call:
        function_name = message.function_call.name
        arguments = json.loads(message.function_call.arguments)
        print(
            f"🧠 Model decided to call function: {function_name} with args {arguments}"
        )

        if function_name == "get_song_by_name":
            result = get_song_by_name(**arguments)
        elif function_name == "get_songs_by_artist":
            result = get_songs_by_artist(**arguments)
        elif function_name == "query_by_attributes":
            result = query_by_attributes(**arguments)
        else:
            result = ["❌ Unknown function"]

        print("🎧 Function call result:", result)
        return result
    else:
        print("🤷 No function call was made. Model response:")
        print(message.content)
        return [message.content]


if __name__ == "__main__":
    user_prompt = "I want ato listen something romantic"
    handle_user_prompt(user_prompt)
