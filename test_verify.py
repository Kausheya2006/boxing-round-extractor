import sys
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ensure the root directory is in the path to import app correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from app.config import VERIFY_MOMENTS_MODEL, VERIFY_MOMENTS_PROMPT

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_verify.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: File not found - {video_path}")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    print(f"Connecting to Gemini API using model: {VERIFY_MOMENTS_MODEL}")
    client = genai.Client(api_key=api_key)

    print(f"Uploading {video_path}...")
    video_file = client.files.upload(file=video_path)
    
    print(f"Uploaded as {video_file.name}. Waiting for processing...")
    while hasattr(video_file, 'state') and str(video_file.state) == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
        
    if hasattr(video_file, 'state') and str(video_file.state) == "FAILED":
        print(f"Video processing failed for {video_file.name}")
        sys.exit(1)
        
    print("Video processed. Running inference... (This may take up to a minute, no need to press any keys)")
    video_part = types.Part(
        file_data=types.FileData(
            file_uri=video_file.uri, 
            mime_type=video_file.mime_type
        ),
        video_metadata=types.VideoMetadata(fps=16.0) 
    )

    generation_config = {
        'max_output_tokens': 65536,
        'temperature': 0.1
    }

    try:
        response = client.models.generate_content(
            model=VERIFY_MOMENTS_MODEL,
            contents=[video_part, VERIFY_MOMENTS_PROMPT],
            config=generation_config,
        )
        print("\n" + "="*40)
        print("         AI RESPONSE")
        print("="*40)
        print(response.text)
        print("="*40 + "\n")
    except Exception as e:
        print(f"\nGeneration failed: {e}")
    finally:
        print("Cleaning up file from Gemini servers...")
        try:
            client.files.delete(name=video_file.name)
            print("Cleanup successful.")
        except Exception as e:
            print(f"Cleanup failed: {e}")

if __name__ == "__main__":
    main()
