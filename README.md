# Auto Label — Video to YOLO Dataset

A Python tool that **automatically builds labeled object-detection datasets from videos**. It samples frames, detects objects with **YOLOv8 (80 COCO classes)** and optionally **YOLO-World** (open-vocabulary, any class you type), then exports YOLO-format labels.

Built to cut down the time spent on manual labeling when preparing custom training data.

## Features

- **Two detectors combined**: COCO model for common objects + YOLO-World for custom classes (e.g. `"hardhat, drill, red backpack"`)
- **Overlapping detections merged** with IoU-based deduplication
- **Per-class duplicate-frame filtering**: a new frame is skipped if it looks almost identical (histogram similarity, default 0.97) to the last saved frame of that class
- **Per-video output folders**, so running several videos never mixes results
- **YOLO-format `.txt` labels**, a `classes.txt` file and a `detections.csv` summary report
- Frames with no detections are saved separately in `unlabeled/`

## Output structure

```
output/
└── video_name/
    ├── frames/            # all sampled raw frames
    ├── labeled_frames/
    │   ├── person/        # frames with person boxes drawn
    │   └── ...
    ├── unlabeled/         # frames with no detections
    ├── labels/            # YOLO-format labels
    ├── classes.txt        # class id -> name
    └── detections.csv     # frame, class, confidence, source model
```

## Installation

```bash
pip install ultralytics opencv-python
```

## Usage

COCO classes only:

```bash
python auto_label_video.py --video cafe.mp4 --output out
```

COCO + YOLO-World with custom classes:

```bash
python auto_label_video.py --video cafe.mp4 --output out \
    --use-world --world-classes "hardhat,drill,red backpack"
```

Read custom classes from a file:

```bash
python auto_label_video.py --video cafe.mp4 --output out \
    --use-world --world-classes-file vocab.txt
```

### Main options

| Option | Default | Description |
|---|---|---|
| `--frame-interval-sec` | `1.0` | Seconds between sampled frames |
| `--conf` | `0.4` | Min confidence for the COCO model |
| `--world-conf` | `0.15` | Min confidence for YOLO-World |
| `--dedup-threshold` | `0.97` | Similarity threshold for skipping duplicate frames |
| `--disable-dedup` | – | Turn off duplicate filtering |
| `--overwrite` | – | Overwrite an existing output folder |

## Tech stack

Python · Ultralytics YOLOv8 · YOLO-World · OpenCV
