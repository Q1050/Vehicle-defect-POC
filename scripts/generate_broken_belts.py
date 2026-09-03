import os
import base64
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

OUTPUT_DIR = Path("synthetic_dataset/broken_belt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

PROMPTS = [
    """
    Create a realistic automotive maintenance photograph showing a clearly damaged
    serpentine belt inside a real car engine bay. The belt should have visible deep
    cracks and wear. Natural workshop lighting, realistic vehicle components,
    documentary repair-photo style, no text, no labels, no arrows.
    """,

    """
    Create a realistic close-up photograph of a car serpentine belt that is badly
    frayed along its edges while still installed around engine pulleys. Realistic
    engine bay, grease and normal mechanical wear, professional mechanic inspection
    photo, no text, no annotations.
    """,

    """
    Create a realistic automotive repair photograph showing a partially torn
    serpentine belt in an engine bay. Several belt ribs should be damaged or missing.
    Keep the belt clearly visible and distinguishable from surrounding engine parts.
    No text, labels, arrows, or diagrams.
    """,

    """
    Create a realistic photograph of a snapped automotive serpentine belt inside
    a vehicle engine compartment. Show the broken ends of the belt near the pulleys.
    Natural lighting, realistic engine components, maintenance documentation style,
    no people, no text, no annotations.
    """,

    """
    Create a realistic close-up vehicle maintenance image showing an old serpentine
    belt with severe cracking, chunking, and material deterioration. The damaged belt
    must be visually obvious. Real car engine bay, photographic realism, no labels
    or text.
    """,

    """
    Create a realistic photograph of an automotive drive belt suffering layer
    separation and fraying while installed in an engine bay. The defect should be
    obvious enough for a mechanic to identify. Realistic workshop photography,
    no text or artificial markings.
    """,
]

IMAGES_PER_PROMPT = 5


def generate_image(prompt, output_path):
    interaction = client.interactions.create(
        model="gemini-3.1-flash-image",
        input=prompt,
    )

    if not interaction.output_image:
        print("No image returned")
        return False

    image_bytes = base64.b64decode(
        interaction.output_image.data
    )

    output_path.write_bytes(image_bytes)

    return True


def main():
    image_number = 1

    for prompt_index, prompt in enumerate(PROMPTS, start=1):

        for variation in range(IMAGES_PER_PROMPT):

            filename = (
                f"broken_belt_synthetic_"
                f"{image_number:03d}.png"
            )

            output_path = OUTPUT_DIR / filename

            print(
                f"Generating {image_number}: "
                f"prompt {prompt_index}, "
                f"variation {variation + 1}"
            )

            try:
                success = generate_image(
                    prompt,
                    output_path
                )

                if success:
                    print(f"Saved: {output_path}")

            except Exception as exc:
                print(f"Generation failed: {exc}")

            image_number += 1

            # Avoid hammering the API.
            time.sleep(2)


if __name__ == "__main__":
    main()