# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prepare CelebAHairMask-HQ, SynthHair and Figaro1k for binary semantic segmentation."""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

EXPECTED_COUNTS = {
    ("train", "CelebAHairMask-HQ"): 29000,
    ("val", "CelebAHairMask-HQ"): 1000,
    ("train", "SynthHair"): 14500,
    ("test_synthhair", "SynthHair"): 1500,
    ("test_figaro1k", "Figaro1k"): 1050,
}
MASK_IDS = [0] * 128 + [1] * 128


def source_pairs(source):
    """Read the source indexes and assign a deterministic real-image validation split."""
    for dataset, index in (("CelebAHairMask-HQ", "pairs.csv"), ("Figaro1k", "test_pairs.csv")):
        with (source / dataset / index).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        is_celeb = dataset == "CelebAHairMask-HQ"
        val_ids = set(random.Random(42).sample(sorted(row["id"] for row in rows), 1000)) if is_celeb else set()
        for row in rows:
            split = ("val" if row["id"] in val_ids else "train") if is_celeb else "test_figaro1k"
            yield split, dataset, row["id"], Path(dataset, row["image"]), Path(dataset, row["mask"])

    dataset = "SynthHair"
    for official_split, split in (("train", "train"), ("test", "test_synthhair")):
        ids = json.loads((source / dataset / f"split_{official_split}.json").read_text(encoding="utf-8"))
        for sample_id in ids:
            name = f"{sample_id:07d}.png"
            yield split, dataset, sample_id, Path(dataset, "image", name), Path(dataset, "scalp_hair_matte", name)


def prepare_pair(pair, source, target):
    """Link a source image and save its aligned mask as an 8-bit PNG of class IDs."""
    split, dataset, sample_id, source_image, source_mask = pair
    if int(source_image.stem) != int(sample_id) or int(source_mask.stem) != int(sample_id):
        raise ValueError(f"Sample ID mismatch for {source_image} and {source_mask}.")
    image = Path("images", split, dataset, source_image.name)
    mask = Path("masks", split, dataset, source_image.name).with_suffix(".png")
    with Image.open(source / source_image) as original, Image.open(source / source_mask) as label:
        if original.size != label.size or label.mode not in {"1", "L", "RGB"}:
            raise ValueError(f"Expected an aligned grayscale mask for {source_image}, got {label.size} {label.mode}.")
        original.verify()
        label.convert("L").point(MASK_IDS).save(target / mask, compress_level=1)
    (target / image).symlink_to(source / source_image)
    return dataset, sample_id, split, image.as_posix(), mask.as_posix(), source_image.as_posix(), source_mask.as_posix()


def main():
    """Build the dataset and publish its manifests after every pair has been prepared."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the three original datasets.")
    parser.add_argument("output", type=Path, help="New directory for the prepared dataset.")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    pairs = list(source_pairs(source))
    counts = Counter((split, dataset) for split, dataset, *_ in pairs)
    if counts != EXPECTED_COUNTS or len({image for _, _, _, image, _ in pairs}) != len(pairs):
        raise ValueError(f"Unexpected split counts or duplicate images: {dict(counts)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        target = Path(tmp) / "dataset"
        for split, dataset in counts:
            for folder in ("images", "masks"):
                (target / folder / split / dataset).mkdir(parents=True)

        images = defaultdict(list)
        with (target / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dataset", "id", "split", "image", "mask", "source_image", "source_mask"))
            with ThreadPoolExecutor(args.workers) as pool:
                for i, row in enumerate(pool.map(partial(prepare_pair, source=source, target=target), pairs), 1):
                    writer.writerow(row)
                    images[row[2]].append(f"./{row[3]}\n")
                    if i % 5000 == 0:
                        print(f"Prepared {i}/{len(pairs)} pairs", flush=True)

        for split, paths in images.items():
            (target / f"{split}.txt").write_text("".join(paths), encoding="utf-8")
        config = "train: train.txt\nval: val.txt\nmasks_dir: masks\nnames: [hair]\n"
        for file, split in (("dataset.yaml", "test_synthhair"), ("dataset_figaro1k.yaml", "test_figaro1k")):
            (target / file).write_text(f"{config}test: {split}.txt\n", encoding="utf-8")
        target.rename(output)
    print(f"Prepared {len(pairs)} pairs in {output}: {dict(counts)}")


if __name__ == "__main__":
    main()
