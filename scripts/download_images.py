import os
import re
import time
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


OUTPUT_DIR = Path("auto_dataset/raw")

IMAGES_PER_QUERY = 30

SEARCH_QUERIES = {
    "broken_belt": [
        "broken automotive serpentine belt",
        "damaged car drive belt",
        "cracked serpentine belt",
        "worn automotive belt",
    ],

    "dashboard_indicator": [
        "car dashboard warning lights",
        "vehicle dashboard warning indicator",
        "check engine light dashboard",
        "car oil warning light",
        "tire pressure warning light dashboard",
    ],

    "oil_leak": [
        "car engine oil leak",
        "vehicle oil leak engine",
        "oil leaking underneath car",
        "automobile engine oil leak",
    ],

    "rust": [
        "car rust corrosion",
        "rusty car body",
        "vehicle rust wheel arch",
        "automobile corrosion",
        "car underbody rust",
    ],

    "tire_wear": [
        "worn car tire tread",
        "bald automobile tire",
        "damaged tire tread",
        "uneven tire wear",
        "worn vehicle tire",
    ],
}


COMMONS_API = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "VehicleDefectResearchDataset/1.0"
}


def clean_filename(name):
    name = re.sub(r"[^\w\-.]", "_", name)
    return name[:150]


def search_commons(query, limit=30):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "format": "json",
    }

    response = requests.get(
        COMMONS_API,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    results = []

    for page in pages.values():
        imageinfo = page.get("imageinfo")

        if not imageinfo:
            continue

        info = imageinfo[0]

        mime = info.get("mime", "")

        if mime not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            continue

        results.append(
            {
                "title": page.get("title", ""),
                "url": info.get("url"),
                "mime": mime,
            }
        )

    return results


def download_image(url, destination):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Avoid very small images.
    if image.width < 300 or image.height < 300:
        return False

    image.save(destination, format="JPEG", quality=95)

    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name, queries in SEARCH_QUERIES.items():

        class_dir = OUTPUT_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {class_name} ===")

        existing = len(list(class_dir.glob("*.jpg")))

        image_index = existing

        seen_urls = set()

        for query in queries:

            print(f"Searching: {query}")

            try:
                results = search_commons(
                    query,
                    limit=IMAGES_PER_QUERY,
                )

            except Exception as exc:
                print(f"Search failed: {exc}")
                continue

            for result in tqdm(results):

                url = result["url"]

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                filename = (
                    f"{class_name}_"
                    f"{image_index:04d}_"
                    f"{clean_filename(result['title'])}.jpg"
                )

                destination = class_dir / filename

                try:
                    if download_image(url, destination):
                        image_index += 1

                except Exception as exc:
                    print(
                        f"\nSkipping {url}: {exc}"
                    )

                time.sleep(0.15)

        print(
            f"{class_name}: "
            f"{len(list(class_dir.glob('*.jpg')))} images"
        )


if __name__ == "__main__":
    main()