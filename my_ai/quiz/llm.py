
import requests
import json
import textwrap
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
# 🗝️ Groq API config
GROQ_API_KEY = "REMOVED_SECRET"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 🧠 System Prompt
system_prompt = """
You are an expert tutor and quiz creator.
Create multiple-choice questions (A–D) from the provided keywords and content summary.
- 70% factual, 30% conceptual or application-based.
- Avoid repetition.
- Each question must be unique, clear, and unambiguous.
Return ONLY valid JSON:
[{"question":"...","options":["A","B","C","D"],"correctIndex":2}, ...]
"""

# 🧹 Clean text
def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 🧩 Extract keywords per chunk
def extract_keywords_tfidf(chunk: str, top_n: int = 100):
    cleaned = clean_text(chunk)
    segments = [s for s in cleaned.split('.') if s.strip()]
    if not segments:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_df=1.0, min_df=1)
    X = vectorizer.fit_transform(segments)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.mean(axis=0).A1
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [word for word, scores in ranked[:top_n]]

# ✂️ Chunk text
def chunk_text(text, max_chars=6000):
    return textwrap.wrap(text, max_chars, break_long_words=False, replace_whitespace=False)

# 🚀 Call Groq API
def call_groq_api(chunk, keywords):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    keyword_context = ", ".join(keywords[:])
    data = {
        "model": "llama-3.3-70B-versatile",
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"Focus on these keywords:\n{keyword_context}\nGenerate unique questions around them"}
        ],
        "temperature": 0.7
    }

    response = requests.post(GROQ_URL, headers=headers, json=data)
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    start, end = cleaned.find("["), cleaned.rfind("]") + 1
    return json.loads(cleaned[start:end])

# 🧠 Main quiz generator
def generate_quiz(content: str, total_questions: int = 40, sleep_time: int = 3):
    print("🚀 Generating quiz...")
    chunks = chunk_text(content)
    questions_per_chunk = max(5, total_questions // len(chunks))

    all_questions = []
    for i, chunk in enumerate(chunks, 1):
        print(f"📚 Processing chunk {i}/{len(chunks)}...")
        keywords = extract_keywords_tfidf(chunk)
        print(f"🧩 Extracted Keywords (chunk {i}): {', '.join(keywords[:10])} ...")
        try:
            qs = call_groq_api(chunk, keywords)
            all_questions.extend(qs)
        except Exception as e:
            print(f"⚠️  Error on chunk {i}: {e}")
            continue
        time.sleep(sleep_time)  # avoid rate-limit errors

    # Deduplicate
    # seen = set()
    # unique_questions = []
    # for q in all_questions:
    #     if q["question"] not in seen:
    #         seen.add(q["question"])
    #         unique_questions.append(q)

    # final = unique_questions[:total_questions]
    print(f"✅ Generated {len(all_questions)} unique questions total.")
    return all_questions


def display_quiz(questions: list):
    """
    Display quiz questions in readable format.
    """
    for q in questions:
        print(q["question"])
        for i, opt in enumerate(q["options"]):
            print(f"  {chr(65+i)}. {opt}")
        print(f"✅ Correct: {chr(65 + q['correctIndex'])}\n")

def init():
    """
    Initializes and runs the quiz generation process.
    """
    sample_text = (
        "Photosynthesis is the process by which green plants and some other organisms use sunlight "
        "to synthesize foods from carbon dioxide and water. Photosynthesis in plants generally involves "
        "the green pigment chlorophyll and generates oxygen as a by-product."
    )
    quiz = generate_quiz(sample_text)
    display_quiz(quiz)

if __name__ == "__main__":
    init()
