import os
import threading
import uuid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .youtube import YouTubeTranscriber
from .pdf import text_from_pdf
from .llm import generate_quiz  # mock or real LLM quiz generation
from .books import generate_chunks
# Thread lock to serialize transcription calls (Whisper + yt-dlp are not guaranteed thread-safe)
transcription_lock = threading.Lock()

# Initialize transcriber once for efficiency (loads Whisper model once)
transcriber = YouTubeTranscriber( )

@api_view(["POST"])
def generate_quiz_from_url(request):
    url = request.data.get("url")
    question_count = int(request.data.get("questionCount", 15))
    print("🔍 Generating quiz from URL:", url)

    if not url:
        return Response({"error": "Missing 'url'"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        if "youtube.com" in url or "youtu.be" in url:
            # Use threading lock to avoid concurrent transcription conflicts
            with transcription_lock:
                content = transcriber.transcribe_youtube_video(url)
        else:
            content = text_from_pdf(url)

        if not content or len(content) < 100:
            raise Exception("Content extraction failed or content too short.")

        questions = generate_quiz(content, question_count)
        return Response({"questions": questions})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
@api_view(["POST"])
def generate_book_chunks_from_url(request):
    book_title = request.data.get("bookTitle")
    start_index = int(request.data.get("startIndex", 0))
    chunk_count = int(request.data.get("count", 5))  # default 5 chunks per request

    print(f"🔍 Generating chunks for '{book_title}', starting at index {start_index}")

    if not book_title:
        return Response({"error": "Missing 'bookTitle'"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Call your chunk generation function with start_index and chunk_count
        chunks = generate_chunks(book_title, start_index=start_index)

        if not chunks:
            raise Exception("Chunk generation failed or returned empty.")

        return Response({
            "title": book_title,
            "chunks": chunks
        })

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
