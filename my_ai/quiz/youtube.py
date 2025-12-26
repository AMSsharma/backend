import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

class YouTubeTranscriber:
    def __init__(self):
        print("🧠 YouTubeTranscriptAPI initialized (no API key needed).")

    def extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from URL."""
        patterns = [
            r"(?:v=)([a-zA-Z0-9_-]{11})",
            r"youtu\.be/([a-zA-Z0-9_-]{11})"
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError("❌ Invalid YouTube URL, could not extract video ID.")

    def transcribe_youtube_video(self, youtube_url: str) -> str:
        """Fetch transcript directly from YouTube captions (fallback if English not available)."""
        video_id = self.extract_video_id(youtube_url)

        try:
            # 1️⃣ Try English first
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        except NoTranscriptFound:
            print("⚠️ English transcript not found, checking alternatives...")

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                # Try Hindi → English translation
                try:
                    transcript = transcript_list.find_transcript(['hi']).translate('en').fetch()
                    print("🌐 Used Hindi auto-generated captions (translated to English).")
                except Exception:
                    # If translation fails, just fetch Hindi (or any available)
                    print("⚠️ Translation failed, using raw Hindi captions instead.")
                    transcript = transcript_list.find_transcript(['hi']).fetch()

            except Exception as e:
                raise RuntimeError(f"❌ No usable transcripts found for video {video_id}: {str(e)}")

        except TranscriptsDisabled:
            raise RuntimeError("❌ Transcripts are disabled for this video.")

        # Join all captions into plain text
        full_text = " ".join([entry['text'] for entry in transcript])
        print(f"✅ Transcript fetched successfully.{full_text}")
        return full_text
