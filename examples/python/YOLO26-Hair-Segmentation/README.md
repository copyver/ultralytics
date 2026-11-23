# YOLO26s binary hair segmentation

Prepare a source directory with `CelebAHairMask-HQ/pairs.csv`, `SynthHair/{image,scalp_hair_matte,split_train.json,split_test.json}`,
and `Figaro1k/test_pairs.csv` covering all 1,050 pairs. The CSV indexes must contain `id`, `image`, and `mask` columns
with paths relative to their source dataset.

| Split          | CelebAHairMask-HQ | SynthHair | Figaro1k | Total  |
| -------------- | ----------------- | --------- | -------- | ------ |
| Train          | 29,000            | 14,500    |          | 43,500 |
| Validation     | 1,000             |           |          | 1,000  |
| SynthHair test |                   | 1,500     |          | 1,500  |
| Figaro1k test  |                   |           | 1,050    | 1,050  |

CelebAHairMask-HQ validation IDs are sampled once from the sorted IDs with seed 42. SynthHair uses the supplied
official split indexes. The two test sets are evaluated separately after training.

```bash
recipe=examples/python/YOLO26-Hair-Segmentation
python "$recipe/prepare.py" "$source_root" "$prepared_root"
yolo cfg="$recipe/train.yaml" data="$prepared_root/dataset.yaml" device=0
```

Set `source_root` and `prepared_root` to absolute paths. The preparation creates image symlinks and aligned 8-bit
PNG masks with pixel values 0 for background and 1 for hair. It thresholds the grayscale labels at 128, including
the SynthHair scalp hair alpha. `manifest.csv` records each split and the original and prepared paths. The four
text lists are consumed by the Ultralytics semantic dataset loader.

Evaluate the trained checkpoint on each test set at 1024 pixels.

```bash
yolo semantic val model="$checkpoint" data="$prepared_root/dataset.yaml" split=test imgsz=1024 device=0
yolo semantic val model="$checkpoint" data="$prepared_root/dataset_figaro1k.yaml" split=test imgsz=1024 device=0
```
