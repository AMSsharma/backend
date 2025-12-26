import json
import pandas as pd
import os

# Load your JSON file
with open("data/difficulty_map.json", "r") as f:
    data = json.load(f)

rows = []
for entry in data:
    difficulty = entry["difficulty"]
    topic = entry["topic"]
    for kw in entry["keywords"]:
        rows.append({
            "text": topic + " " + kw,   # combine topic + keyword (optional)
            "difficulty": difficulty
        })

df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("difficulty_dataset.csv", index=False)

print(df.head(15))
