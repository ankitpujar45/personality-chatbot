import csv
import random

TARGET_SIZE = 500
INPUT_FILE = "ground_truth_sample.csv"
OUTPUT_FILE = "ground_truth_sample.csv"

paraphrases = [
    "today",
    "right now",
    "these days",
    "lately",
    "at the moment",
    "recently",
    
]

rows = []

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

expanded = []
id_counter = 1

while len(expanded) < TARGET_SIZE:
    base = random.choice(rows)
    phrase = random.choice(paraphrases)

    text = base["text"]
    if phrase and phrase not in text.lower():
        text = text + " " + phrase

    expanded.append({
        "id": id_counter,
        "text": text.strip(),
        "ground_truth": base["ground_truth"]
    })

    id_counter += 1

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["id", "text", "ground_truth"]
    )
    writer.writeheader()
    writer.writerows(expanded)

print(f"✅ Generated {TARGET_SIZE} ground truth samples in {OUTPUT_FILE}")
