from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re


def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL.")


def get_transcript(url: str) -> str:
    """
    Fetch the transcript for a given YouTube URL and return it as a single
    cleaned string.
    """
    video_id = extract_video_id(url)
    ytt_api = YouTubeTranscriptApi()

    try:
        fetched_transcript = ytt_api.fetch(video_id)
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise ValueError("No transcript found for this video.")

    full_text = " ".join(snippet.text for snippet in fetched_transcript)
    return full_text


if __name__ == "__main__":
    test_url = input("Enter a YouTube URL: ")
    try:
        text = get_transcript(test_url)
        print(f"\nTranscript length: {len(text)} characters\n")
        print(text[:500], "...")
    except ValueError as e:
        print(f"Error: {e}")