from google import genai
import os

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "Missing GEMINI_API_KEY environment variable. "
    )

client = genai.Client(api_key=api_key)