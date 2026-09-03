from pathlib import Path
from PIL import Image

INPUT_DIR = Path("synthetic_belt_sheets")
OUTPUT_DIR = Path("synthetic_dataset/broken_belt")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure each contact sheet manually:
# filename: (rows, cols)

SHEETS = {
    "belt_sheet_1.png": (5, 5),   # 25 images
    "belt_sheet_2.png": (5, 5),   # 25 images
    "belt_sheet_3.png": (2, 5),   # 10 images
}


def split_sheet(image_path, rows, cols, start_index):
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    tile_width = width // cols
    tile_height = height // rows

    index = start_index

    for row in range(rows):
        for col in range(cols):

            left = col * tile_width
            top = row * tile_height

            right = (
                width
                if col == cols - 1
                else (col + 1) * tile_width
            )

            bottom = (
                height
                if row == rows - 1
                else (row + 1) * tile_height
            )

            tile = image.crop(
                (left, top, right, bottom)
            )

            output_path = (
                OUTPUT_DIR
                / f"broken_belt_synthetic_{index:03d}.jpg"
            )

            tile.save(
                output_path,
                "JPEG",
                quality=95,
            )

            print(f"Saved {output_path}")

            index += 1

    return index


def main():
    current_index = 1

    for filename, (rows, cols) in SHEETS.items():

        image_path = INPUT_DIR / filename

        if not image_path.exists():
            print(f"Skipping missing file: {image_path}")
            continue

        print(
            f"\nSplitting {filename}: "
            f"{rows} rows x {cols} columns"
        )

        current_index = split_sheet(
            image_path,
            rows,
            cols,
            current_index,
        )

    print(
        f"\nDone. Created "
        f"{current_index - 1} images."
    )


if __name__ == "__main__":
    main()