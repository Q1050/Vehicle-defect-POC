from pathlib import Path
import shutil
import random
import yaml

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PUBLIC_DATASETS = PROJECT_ROOT / "public_datasets"
OUTPUT_DIR = PROJECT_ROOT / "dataset_merged"

TARGET_CLASSES = {
    "broken_belt": 0,
    "dashboard_indicator": 1,
    "oil_leak": 2,
    "rust": 3,
    "tire_wear": 4,
}

# Each source dataset gets:
# - folder: dataset folder inside public_datasets/
# - class_map: original class name -> final target class name
#
# Update these names after checking each source data.yaml.
SOURCES = [
    {
        "name": "original",
        "folder": PROJECT_ROOT / "dataset",
        "class_map": {
            "broken_belt": "broken_belt",
            "dashboard_indicator": "dashboard_indicator",
            "oil_leak": "oil_leak",
            "rust": "rust",
            "tire_wear": "tire_wear",
        },
    },

{
    "name": "rust",
    "folder": PUBLIC_DATASETS / "rust",
    "class_map": {
        "2_Fair_Steel_Corrosion": "rust",
        "3_Poor_Steel_Corrosion": "rust",
        "4_Severe_Steel_Corrosion": "rust",
    },
},

    {
        "name": "oil_leak",
        "folder": PUBLIC_DATASETS / "oil_leak",
        "class_map": {
            "oil_leak": "oil_leak",
            "oil leak": "oil_leak",
            "leak": "oil_leak",
        },
    },

 {
    "name": "dashboard",
    "folder": PUBLIC_DATASETS / "dashboard",
    "class_map": {
        "Anti Lock Braking System": "dashboard_indicator",
        "Braking System Issue": "dashboard_indicator",
        "Charging System Issue": "dashboard_indicator",
        "Check Engine": "dashboard_indicator",
        "Electronic Stability Problem -ESP-": "dashboard_indicator",
        "Engine Overheating Warning Light": "dashboard_indicator",
        "Low Engine Oil Warning Light": "dashboard_indicator",
        "Low Tire Pressure Warning Light": "dashboard_indicator",
        "Master warning light": "dashboard_indicator",
        "SRS-Airbag": "dashboard_indicator",
    },
},

    {
        "name": "tire",
        "folder": PUBLIC_DATASETS / "tire",
        "class_map": {
            "bad_tyres": "tire_wear",
            "bad_tires": "tire_wear",
            "bald_tyres": "tire_wear",
            "bald_tires": "tire_wear",
            "worn_tire": "tire_wear",
            "tire_wear": "tire_wear",
            "BAD_Tyres": "tire_wear",
            "BALD_Tyres": "tire_wear",

            # Healthy tire classes intentionally NOT mapped.
        },
    },
    {
    "name": "broken_belt",
    "folder": PUBLIC_DATASETS / "broken_belt",
    "class_map": {
        "broken_belt": "broken_belt",
    },
},
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

random.seed(42)


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def normalize_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_yaml(dataset_folder):
    yaml_path = dataset_folder / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No data.yaml found in {dataset_folder}"
        )

    with yaml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_names(data):
    names = data.get("names", {})

    if isinstance(names, list):
        return {
            index: name
            for index, name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(index): name
            for index, name in names.items()
        }

    raise ValueError("Unsupported names format in data.yaml")


def find_split_folder(dataset_folder, split):
    candidates = {
        "train": ["train"],
        "valid": ["valid", "val"],
        "test": ["test"],
    }

    for candidate in candidates[split]:
        path = dataset_folder / candidate

        if path.exists():
            return path

    return None


def get_label_path(image_path, split_folder):
    image_dir = split_folder / "images"
    label_dir = split_folder / "labels"

    relative = image_path.relative_to(image_dir)

    return (
        label_dir
        / relative
    ).with_suffix(".txt")


def rewrite_label_file(
    label_path,
    original_names,
    class_map,
):
    if not label_path.exists():
        return []

    output_lines = []

    normalized_map = {
        normalize_name(source): target
        for source, target in class_map.items()
    }

    with label_path.open("r", encoding="utf-8") as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            old_class_id = int(float(parts[0]))

            original_class_name = original_names.get(
                old_class_id
            )

            if original_class_name is None:
                continue

            normalized_original = normalize_name(
                original_class_name
            )

            target_name = normalized_map.get(
                normalized_original
            )

            if not target_name:
                continue

            new_class_id = TARGET_CLASSES[target_name]

            output_lines.append(
                " ".join(
                    [str(new_class_id)] + parts[1:]
                )
            )

    return output_lines


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


def copy_dataset_source(source):

    dataset_folder = source["folder"]

    if not dataset_folder.exists():

        print(
            f"\nSkipping {source['name']}: "
            f"{dataset_folder} does not exist"
        )

        return {}

    data = load_yaml(dataset_folder)

    original_names = extract_names(data)

    print(
        f"\n=== SOURCE: {source['name']} ==="
    )

    print("Classes found:")

    for class_id, class_name in original_names.items():

        print(
            f"  {class_id}: {class_name}"
        )

    counts = {
        name: 0
        for name in TARGET_CLASSES
    }

    for split in ["train", "valid", "test"]:

        split_folder = find_split_folder(
            dataset_folder,
            split,
        )

        if split_folder is None:
            continue

        image_dir = split_folder / "images"

        if not image_dir.exists():
            continue

        images = [
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        for image_path in images:

            label_path = get_label_path(
                image_path,
                split_folder,
            )

            rewritten_labels = rewrite_label_file(
                label_path,
                original_names,
                source["class_map"],
            )

            # Skip images where none of the wanted classes exist.
            if not rewritten_labels:
                continue

            safe_name = (
                f"{source['name']}_"
                f"{split}_"
                f"{image_path.stem}"
                f"{image_path.suffix.lower()}"
            )

            destination_image = (
                OUTPUT_DIR
                / split
                / "images"
                / safe_name
            )

            destination_label = (
                OUTPUT_DIR
                / split
                / "labels"
                / f"{Path(safe_name).stem}.txt"
            )

            shutil.copy2(
                image_path,
                destination_image,
            )

            destination_label.write_text(
                "\n".join(rewritten_labels),
                encoding="utf-8",
            )

            for line in rewritten_labels:

                class_id = int(
                    line.split()[0]
                )

                class_name = next(
                    name
                    for name, target_id
                    in TARGET_CLASSES.items()
                    if target_id == class_id
                )

                counts[class_name] += 1

    return counts


def write_final_yaml():

    data = {
        "path": str(
            OUTPUT_DIR.resolve()
        ),

        "train": "train/images",

        "val": "valid/images",

        "test": "test/images",

        "nc": len(TARGET_CLASSES),

        "names": {
            class_id: class_name
            for class_name, class_id
            in TARGET_CLASSES.items()
        },
    }

    with (
        OUTPUT_DIR / "data.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
        )


def count_images():

    print("\n=== FINAL IMAGE COUNTS ===")

    total = 0

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

        count = len(
            [
                path
                for path in image_dir.glob("*")
                if path.suffix.lower()
                in IMAGE_EXTENSIONS
            ]
        )

        total += count

        print(
            f"{split:8}: {count}"
        )

    print(
        f"total   : {total}"
    )


def main():

    prepare_output()

    total_counts = {
        name: 0
        for name in TARGET_CLASSES
    }

    for source in SOURCES:

        try:

            counts = copy_dataset_source(
                source
            )

        except Exception as exc:

            print(
                f"Failed processing "
                f"{source['name']}: "
                f"{exc}"
            )

            continue

        for class_name, count in counts.items():

            total_counts[class_name] += count

    write_final_yaml()

    print(
        "\n=== FINAL ANNOTATION COUNTS ==="
    )

    for class_name, count in total_counts.items():

        print(
            f"{class_name:22} "
            f"{count}"
        )

    count_images()

    print(
        "\nMerged YOLO dataset:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main() 