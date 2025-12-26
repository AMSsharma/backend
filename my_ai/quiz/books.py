import requests
import json

# 🗝️ Groq API config
GROQ_API_KEY = "GROQ_API_KEY"
GROQ_API_URL = "GROQ_API_URL"
# 🧠 System prompt for generating book chunks
system_prompt = """
You are an expert tutor and book summarizer.

Given the book title provided by the user, divide its main learnings and stories into distinct learning chunks.

Requirements:
- Divide the book into 20 chunks (if possible).
- Each chunk should be distinct and cover a different theme, story, or framework from the book. Avoid repeating the same keyPoints in multiple chunks unless absolutely necessary.
- Distribute content evenly across chunks to cover the full scope of the book.

Each chunk must follow this exact JSON format:
{
  "chunkIndex": <number>,
  "chunkText": {
      "keyPoints": [
          "At least 10 unique insights or key lessons in one sentence each.",
          "Avoid repeating keyPoints already used in earlier chunks."
      ],
      "realLifeExamples": [
          "**Example 1:** A real-world scenario that illustrates the above points.",
          "**Example 2:** Another practical scenario."
      ],
      "takeaway": "**A single, simple takeaway in plain words for quick recall.**"
  }
}

Output rules:
- Each chunk must contain ≥10 keyPoints.
- Each chunk must contain exactly 2 bold real-life examples.
- Each chunk must contain exactly 1 bold takeaway.
- Respond ONLY with a valid JSON array. No markdown fences, no text outside JSON.
"""

def generate_chunks(book_title: str, start_index: int = 0, count: int = 20) -> list:
    """
    Generate learning chunks for a given book using Groq API.

    Args:
        book_title (str): Name of the book.
        start_index (int): Starting chunk index (for batch fetching).
        count (int): Number of chunks to generate in this batch.

    Returns:
        list: List of JSON objects representing chunks.
    """
    print(f"🚀 Generating {count} chunks for book: {book_title}, starting at index {start_index}")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    user_prompt = f"Book title: {book_title}\nGenerate chunks {start_index + 1} to {start_index + count}."

    data = {
        "model": "llama-3.3-70B-versatile",
        "messages": [
            {"role": "system", "content": system_prompt.strip().replace("{count}", str(count))},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(GROQ_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    raw = result["choices"][0]["message"]["content"]
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    # Extract JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]") + 1
    json_chunk = cleaned[start:end]

    try:
        chunks = json.loads(json_chunk)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse extracted JSON:", e)
        print("Extracted JSON:", json_chunk)
        raise

    if not isinstance(chunks, list):
        raise ValueError("Expected a list of chunks.")

    if len(chunks) != count:
        print(f"⚠️  Warning: Expected {count} chunks, got {len(chunks)}.")

    print("🎉 Chunks generated successfully!")
    print(chunks)
    return chunks
