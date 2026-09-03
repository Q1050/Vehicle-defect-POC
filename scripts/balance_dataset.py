from pathlib import Path
import random
import shutil
from collections import defaultdict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "dataset_merged"
OUTPUT_DIR = PROJECT_ROOT / "dataset_balanced"

TARGET_CLASSES = {
    0: "broken_belt",
    1: "dashboard_indicator",
    2: "oil_leak",
    3: "rust",
    4: "tire_wear",
}

# Approximate image targets per class.
# Broken belt is intentionally left small because data is limited.
TARGET_IMAGES = {
    0: 63,     # broken_belt
    1: 350,    # dashboard_indicator
    2: 300,    # oil_leak
    3: 350,    # rust
    4: 350,    # tire_wear
}

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

random.seed(42)


def get_images():
    images = []

    for split in ["train", "valid", "test"]:
        image_dir = SOURCE_DIR / split / "images"
        label_dir = SOURCE_DIR / split / "labels"

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            images.append((image_path, label_path))

    return images


def get_classes_from_label(label_path):
    class_ids = set()

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(float(parts[0]))
                class_ids.add(class_id)
            except ValueError:
                continue

    return class_ids


def build_class_index(images):
    class_to_images = defaultdict(list)

    for image_path, label_path in images:
        class_ids = get_classes_from_label(label_path)

        for class_id in class_ids:
            class_to_images[class_id].append(
                (image_path, label_path)
            )

    return class_to_images


def select_images(class_to_images):
    selected = set()

    # Start with minority classes first.
    priority = [
        0,  # broken_belt
        2,  # oil_leak
        4,  # tire_wear
        3,  # rust
        1,  # dashboard
    ]

    for class_id in priority:
        candidates = class_to_images.get(class_id, [])

        random.shuffle(candidates)

        target = TARGET_IMAGES[class_id]

        chosen = candidates[:target]

        for image_path, label_path in chosen:
            selected.add(
                (image_path, label_path)
            )

        print(
            f"{TARGET_CLASSES[class_id]:22} "
            f"available={len(candidates):4} "
            f"selected={len(chosen):4}"
        )

    return list(selected)


def prepare_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "valid", "test"]:
        (OUTPUT_DIR / split / "images").mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DIR / split / "labels").mkdir(
            parents=True,
            exist_ok=True,
        )


def ensure_all_classes_in_eval(
    selected_images,
    val_images,
    test_images,
):
    """
    Try to ensure every class appears at least once
    in validation and test.
    """

    class_candidates = defaultdict(list)

    for item in selected_images:
        _, label_path = item

        for class_id in get_classes_from_label(label_path):
            class_candidates[class_id].append(item)

    def add_missing(split_images):
        split_set = set(split_images)

        present = set()

        for _, label_path in split_images:
            present.update(
                get_classes_from_label(label_path)
            )

        for class_id in TARGET_CLASSES:
            if class_id in present:
                continue

            candidates = class_candidates.get(
                class_id,
                []
            )

            for candidate in candidates:
                if candidate not in split_set:
                    split_images.append(candidate)
                    split_set.add(candidate)
                    break

    add_missing(val_images)
    add_missing(test_images)


def split_dataset(selected_images):

    synthetic_belts = []
    real_images = []

    for item in selected_images:
        image_path, _ = item

        # Anything coming from public_datasets/broken_belt
        # was generated synthetically.
        if image_path.name.startswith("broken_belt_"):
            synthetic_belts.append(item)
        else:
            real_images.append(item)

    print(
        f"\nSynthetic broken-belt images forced to TRAIN: "
        f"{len(synthetic_belts)}"
    )

    random.shuffle(real_images)

    total = len(real_images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = real_images[:train_end]
    val_images = real_images[train_end:val_end]
    test_images = real_images[val_end:]

    # Synthetic images are training-only.
    train_images.extend(synthetic_belts)

    return (
        train_images,
        val_images,
        test_images,
    )


def copy_split(split_name, items):
    used_names = set()

    for index, (image_path, label_path) in enumerate(
        items,
        start=1,
    ):
        suffix = image_path.suffix.lower()

        base_name = image_path.stem

        safe_name = base_name

        if safe_name in used_names:
            safe_name = f"{base_name}_{index}"

        used_names.add(safe_name)

        destination_image = (
            OUTPUT_DIR
            / split_name
            / "images"
            / f"{safe_name}{suffix}"
        )

        destination_label = (
            OUTPUT_DIR
            / split_name
            / "labels"
            / f"{safe_name}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image,
        )

        shutil.copy2(
            label_path,
            destination_label,
        )


def count_annotations(split_name):
    label_dir = OUTPUT_DIR / split_name / "labels"

    counts = defaultdict(int)

    for label_path in label_dir.glob("*.txt"):
        with label_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(float(parts[0]))

                counts[class_id] += 1

    return counts


def count_images_per_class(split_name):
    label_dir = OUTPUT_DIR / split_name / "labels"

    counts = defaultdict(int)

    for label_path in label_dir.glob("*.txt"):
        class_ids = get_classes_from_label(
            label_path
        )

        for class_id in class_ids:
            counts[class_id] += 1

    return counts


def write_yaml():
    config = {
        "path": str(
            OUTPUT_DIR.resolve()
        ),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
    }

    with (
        OUTPUT_DIR / "data.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


def print_summary():
    print("\n=== FINAL BALANCED DATASET ===")

    for split in [
        "train",
        "valid",
        "test",
    ]:
        image_dir = (
            OUTPUT_DIR
            / split
            / "images"
        )

        image_count = len(
            list(image_dir.iterdir())
        )

        print(
            f"\n{split.upper()} "
            f"({image_count} images)"
        )

        annotation_counts = count_annotations(
            split
        )

        image_class_counts = count_images_per_class(
            split
        )

        for class_id, class_name in TARGET_CLASSES.items():
            print(
                f"{class_name:22} "
                f"images={image_class_counts[class_id]:4} "
                f"boxes={annotation_counts[class_id]:4}"
            )


def main():
    print(
        f"Reading merged dataset from:\n"
        f"{SOURCE_DIR.resolve()}\n"
    )

    images = get_images()

    print(
        f"Found {len(images)} labeled images.\n"
    )

    class_to_images = build_class_index(
        images
    )

    selected_images = select_images(
        class_to_images
    )

    print(
        f"\nUnique selected images: "
        f"{len(selected_images)}"
    )

    prepare_output()

    (
        train_images,
        val_images,
        test_images,
    ) = split_dataset(
        selected_images
    )

    copy_split(
        "train",
        train_images,
    )

    copy_split(
        "valid",
        val_images,
    )

    copy_split(
        "test",
        test_images,
    )

    write_yaml()

    print_summary()

    print(
        f"\nBalanced dataset created at:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()