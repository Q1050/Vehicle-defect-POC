from pathlib import Path
from PIL import Image

INPUT_DIR = Path("synthetic_dataset/broken_belt")
OUTPUT_DIR = Path("synthetic_dataset/broken_belt_clean")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIM_BOTTOM_PERCENT = 0.12

for image_path in INPUT_DIR.glob("*.jpg"):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size
    new_height = int(height * (1 - TRIM_BOTTOM_PERCENT))

    cropped = image.crop((0, 0, width, new_height))

    output_path = OUTPUT_DIR / image_path.name
    cropped.save(output_path, "JPEG", quality=95)

    print(f"Saved {output_path}")