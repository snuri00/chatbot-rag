import json
import sys
import time

import requests

API_URL = "http://localhost:8000"
MET_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects"

HIGHLIGHT_IDS = [
    436535,  # Wheat Field with Cypresses - Van Gogh
    437133,  # The Death of Socrates - Jacques-Louis David
    438012,  # Madonna and Child - Duccio di Buoninsegna
    436532,  # Irises - Van Gogh
    437329,  # Washington Crossing the Delaware - Leutze
    459027,  # The Dance Class - Degas
    438815,  # The Storm - Pierre-Auguste Cot
    436528,  # Self-Portrait with a Straw Hat - Van Gogh
    437984,  # Young Woman with a Water Pitcher - Vermeer
    435809,  # Aristotle with a Bust of Homer - Rembrandt
    438821,  # The Horse Fair - Rosa Bonheur
    436105,  # A Sunday on La Grande Jatte - study
    437854,  # Madame X - John Singer Sargent
    435882,  # The Harvesters - Pieter Bruegel
    436524,  # Cypresses - Van Gogh
]


def fetch_artwork(object_id):
    r = requests.get(f"{MET_API}/{object_id}", timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def build_description(data):
    parts = []

    if data.get("title"):
        parts.append(f'"{data["title"]}"')

    artist = ""
    if data.get("constituents"):
        artist = data["constituents"][0].get("name", "")
        if artist:
            parts.append(f"by {artist}")

    if data.get("objectDate"):
        parts.append(f"dated {data['objectDate']}")

    if data.get("medium"):
        parts.append(f"Medium: {data['medium']}.")

    if data.get("dimensions"):
        parts.append(f"Dimensions: {data['dimensions']}.")

    if data.get("department"):
        parts.append(f"Department: {data['department']}.")

    if data.get("culture"):
        parts.append(f"Culture: {data['culture']}.")

    if data.get("period"):
        parts.append(f"Period: {data['period']}.")

    if data.get("creditLine"):
        parts.append(f"Credit: {data['creditLine']}.")

    if data.get("GalleryNumber"):
        parts.append(f"Gallery Number: {data['GalleryNumber']}.")

    if data.get("repository"):
        parts.append(f"Repository: {data['repository']}.")

    return " ".join(parts)


def ingest(data, description):
    title = data.get("title", "Unknown")
    artist = data["constituents"][0]["name"] if data.get("constituents") else "Unknown"
    image_url = data.get("primaryImageSmall", "")

    payload = {
        "content": description,
        "metadata": {
            "source": "met_museum",
            "title": title,
            "type": "artwork",
            "artist": artist,
            "period": data.get("period", data.get("objectDate", "")),
            "location": f"Gallery {data.get('GalleryNumber', 'N/A')}",
            "language": "en",
            "image_urls": [image_url] if image_url else [],
            "tags": [
                data.get("objectName", ""),
                data.get("department", ""),
                data.get("medium", "").split(",")[0] if data.get("medium") else "",
                artist,
            ],
        },
    }

    r = requests.post(f"{API_URL}/ingest", json=payload, timeout=60)
    return r.json()


def main():
    print(f"Seeding {len(HIGHLIGHT_IDS)} artworks from The Metropolitan Museum of Art...")
    print()

    success = 0
    for i, obj_id in enumerate(HIGHLIGHT_IDS, 1):
        data = fetch_artwork(obj_id)
        if not data or not data.get("title"):
            print(f"  [{i}/{len(HIGHLIGHT_IDS)}] SKIP object {obj_id}")
            continue

        description = build_description(data)
        title = data.get("title", "Unknown")
        artist = data["constituents"][0]["name"] if data.get("constituents") else "Unknown"

        try:
            result = ingest(data, description)
            chunks = result.get("chunk_count", 0)
            print(f"  [{i}/{len(HIGHLIGHT_IDS)}] {title} - {artist} ({chunks} chunks)")
            success += 1
        except Exception as e:
            print(f"  [{i}/{len(HIGHLIGHT_IDS)}] FAIL {title}: {e}")

        time.sleep(0.5)

    print()
    print(f"Done: {success}/{len(HIGHLIGHT_IDS)} artworks ingested.")


if __name__ == "__main__":
    main()
