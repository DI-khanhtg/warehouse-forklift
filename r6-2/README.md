# R6.2 Forklift Cockpit Phone-Use Detection PoC

An explainable baseline that detects whether a forklift operator is using a phone. It combines a pretrained COCO phone detector, a pretrained pose model, pose-assisted high-resolution phone crops, normalized spatial rules, and a sliding temporal filter. No custom training or action-recognition network is used.

## How it works

```text
MP4 / cockpit camera
        |
        v
optional fixed driver ROI
        |
        +-----------------------+
        |                       |
        v                       v
full-frame phone YOLO      YOLO pose
                                |
                         scale-aware ROIs
                   wrists / both hands / head
                                |
                         crop phone YOLO
        |                       |
        +-----------+-----------+
                    v
             IoU deduplication
                    v
       normalized spatial rules
       phone near hand/head?
                  |
                  v
       sliding window + hysteresis
                  |
                  v
       NORMAL / USING_PHONE
                  |
          annotated MP4 + events
```

Phone visibility by itself is **not** a violation. The baseline requires evidence that the phone is near a wrist or head. Distances and search regions are normalized using shoulder width, with person-box/frame fallbacks. `PHONE_CALL` has priority over `TEXTING_OR_HOLDING_PHONE`.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
conda create -n r6_phone python=3.11 -y
conda activate r6_phone
pip install -r requirements.txt
```

Model weights are kept under `weights/`. If a supported YOLO11 weight is missing, Ultralytics downloads it there on first use. CUDA is selected automatically when PyTorch reports an available GPU; otherwise inference uses CPU. For a CUDA-specific PyTorch build, follow the installation command for the machine's CUDA version before installing the requirements.

## Project structure

```text
r6-2/
├── infer_camera.py              # live-camera entrypoint
├── infer_video.py               # video entrypoint
├── debug_phone.py               # detector comparison utility
├── evaluate.py                  # event evaluation utility
├── forklift_phone_detection/    # reusable application package
│   ├── config.py
│   ├── pipeline.py
│   ├── logic/
│   ├── models/
│   └── utils/
├── tests/
├── weights/                     # local YOLO weights
├── input/                       # user-provided media
└── output/                      # generated videos, events, and debug data
```

## Run video

Place a test video in `input/`, then run:

```bash
python infer_video.py --source input/cockpit_test.mp4
```

Default outputs:

```text
output/cockpit_test_annotated.mp4
output/events.csv
output/events.json
output/cockpit_test_run_report.json
```

The default annotated output is HD `1280x720` in `phone_only` mode. It shows accepted `cell phone` boxes, the driver's face/shoulder/elbow/wrist pose skeleton, stable driver track ID/status, `NORMAL`/`USING_PHONE`, FPS, and an evidence line when a phone is near a wrist or head. Search ROI boxes and low-confidence phone boxes remain hidden while the full recall logic continues internally. The source aspect ratio is preserved with letterboxing.

Use a different display/output resolution when needed:

```bash
python infer_video.py --source input/cockpit_test.mp4 --display-size 1920x1080
python infer_video.py --source input/cockpit_test.mp4 --display-mode debug
```

Choose an output path or tune thresholds without editing source:

```bash
python infer_video.py --source input/cockpit_test.mp4 --output output/result.mp4 \
  --phone-model yolo11s.pt --phone-image-size 960 \
  --hand-threshold 0.6 --head-threshold 0.7 --alert-on-ratio 0.6
```

PowerShell uses a backtick instead of `\` for multiline commands. Add `--preview` to display the annotated result while processing. Press `q` to stop the preview.

## Run live camera

```bash
python infer_camera.py --camera 0
```

Press `q` to exit. Camera inference uses exactly the same models, rules, filter, annotations, and event tracker as video inference.

The live window also defaults to HD `1280x720` with the `phone_only` overlay:

```bash
python infer_camera.py --camera 1 --display-size 1280x720
```

On CPU, camera `auto` mode uses `yolo11n` at 640, pose at 320, and compact hands/head crops at 320. Secondary crops run on every processed frame so a previous phone box is never replayed at an old location. CUDA keeps the full recall settings. Explicit CLI quality options such as `--pose-image-size` or `--phone-model` take priority over these automatic defaults. Use either behavior explicitly if needed:

```bash
python infer_camera.py --camera 1 --camera-mode optimized
python infer_camera.py --camera 1 --camera-mode recall
```

`recall` restores `yolo11s` at 960 and all local crops every frame, but measured close to 1 FPS on this CPU.

## Driver ROI

For a fixed cockpit view, crop inference to the operator using normalized coordinates:

```bash
python infer_video.py --source input/cockpit_test.mp4 --roi 0.15,0.05,0.85,1.0
```

The values are `x1,y1,x2,y2`, each from 0 to 1. Start with the whole frame, then configure the ROI after reviewing a representative cockpit frame. A tight operator ROI is strongly recommended for a fixed cockpit camera because it also anchors the first driver track and prevents a larger bystander from being selected at startup.

## Important configuration

Defaults are centralized in `forklift_phone_detection/config.py`:

| Setting | Default | Meaning |
|---|---:|---|
| Phone model | `yolo11s.pt` | Configurable as `yolo11n/s/m.pt` |
| Phone confidence | 0.10 | Candidate acceptance threshold |
| Raw debug confidence | 0.01 | Internal threshold used to expose filtered candidates |
| Full phone image size | 960 | Full-frame/driver-ROI detector resolution |
| Crop phone image size | 640 | Wrist/hands/head crop detector resolution |
| Phone ROI scale | 0.75 | Search radius relative to shoulder width |
| Keypoint confidence | 0.30 | Minimum usable pose-keypoint confidence |
| Driver tracking | enabled | Associate one operator across consecutive frames |
| Pose smoothing alpha | 0.55 | EMA weight for the newest accepted pose |
| Driver max center jump | 0.35 | Reject abrupt bbox jumps, normalized by bbox diagonal |
| Driver max missed frames | 3 | Hold a lost track for display before reacquiring |
| Keypoint max jump | 0.85 | Reject isolated joint jumps, normalized by body scale |
| Hand distance | 0.60 | Phone-to-wrist distance / shoulder width |
| Head distance | 0.70 | Phone-to-ear/head distance / shoulder width |
| Window | 1.5 s | Timestamp-based temporal history duration |
| Alert on | 0.60 | Positive ratio that activates the alert |
| Alert off | 0.30 | Ratio that clears an active alert |

The phone class number is not hardcoded; it is located by searching `model.names` for `cell phone`. Full-frame and local-crop candidates are merged with IoU NMS. Every accepted detection records `full_frame`, `left_wrist_roi`, `right_wrist_roi`, `both_hands_roi`, or `head_roi` as its source. The temporal window uses actual timestamps rather than camera-reported FPS; it must be at least half filled and contain at least two observations before it can alert.

Driver tracking uses the previous operator bbox to select the next pose, rejects implausible person jumps and isolated wrist/elbow jumps, corrects transient left/right arm swaps, and smooths accepted joints. A temporarily missing/rejected pose is drawn in gray as `HELD`, but it is display-only and cannot generate phone-use evidence. When the driver identity changes, temporal votes are cleared so evidence from two different people cannot be combined.

Useful switches:

```text
--phone-conf FLOAT
--phone-image-size 640|960|1280
--phone-crop-image-size 640|960|1280
--phone-roi-scale FLOAT
--phone-nms-iou FLOAT
--debug-phone-detection / --no-debug-phone-detection
--pose-phone-search / --no-pose-phone-search
--show-phone-search-rois / --no-show-phone-search-rois
--phone-debug-log-every N
--phone-crop-interval N
--compact-phone-search-rois / --no-compact-phone-search-rois
--pose-conf FLOAT
--keypoint-conf FLOAT
--driver-tracking / --no-driver-tracking
--pose-smoothing-alpha FLOAT
--driver-max-center-jump FLOAT
--driver-min-iou FLOAT
--driver-max-missed-frames N
--pose-keypoint-max-jump FLOAT
--hand-threshold FLOAT
--head-threshold FLOAT
--window-seconds FLOAT
--alert-on-ratio FLOAT
--alert-off-ratio FLOAT
--image-size INT
--device auto|cpu|0|cuda:0
--allow-phone-outside-body
--require-below-shoulders
--hide-pose
--hide-debug-lines
--display-mode phone_only|debug
```

`--image-size` remains as a legacy shortcut that sets the phone, crop, and pose sizes together. For the requested recall settings, prefer the independent options above.

## Diagnose missed phones

Raw phone debugging is enabled by default. YOLO runs internally at confidence `0.01`, while only candidates at or above `0.10` enter the interaction rules. The result therefore distinguishes `no_candidate`, `low_confidence_filtered`, and `accepted`. Low-confidence boxes and accepted detection sources are drawn only while raw debugging is enabled. Periodic raw-candidate logs are opt-in:

```bash
python infer_video.py --source input/test.mp4 --phone-debug-log-every 30
```

Save sampled original frames where pose sees at least one wrist but no accepted phone is near a wrist:

```bash
python infer_video.py --source input/test.mp4 --save-debug-frames
```

Images and JSON sidecars are written to `output/debug/missed_phone/`. The default interval is one second; change it with `--debug-frame-interval`.

Compare all requested pretrained models and image sizes on one failure image:

```bash
python debug_phone.py --source input/frame.jpg
```

This runs `yolo11n`, `yolo11s`, and `yolo11m` at 640, 960, and 1280, prints the best raw phone confidence table, and writes nine annotated images plus CSV/JSON summaries under `output/debug/phone_comparison/frame/`.

## Event evaluation

Copy `examples/ground_truth.example.json` and manually mark the true `USING_PHONE` intervals. Then run:

```bash
python evaluate.py --predictions output/events.csv \
  --ground-truth examples/ground_truth.example.json \
  --duration 222 --output output/evaluation.json
```

Every video run also saves a JSON summary containing duration, frames, detector counts, detected events, processing FPS, ROI, and the exact thresholds used. The evaluation reports event-level precision, recall, F1, false-positive/false-negative events, false alarms per hour, and average detection delay. Events are matched when their time intervals overlap. These behavior metrics are more relevant to R6.2 than detector mAP.

## Tests

```bash
pytest -q
```

Tests cover geometry, normalized scale, instantaneous rules, sliding windows, hysteresis, merged events, CSV output, and event evaluation. They do not download YOLO weights.

## Manual test checklist

| ID | Scenario | Expected |
|---|---|---|
| T01 | Normal driving | `NORMAL` |
| T02 | Phone held in left hand | `USING_PHONE` |
| T03 | Phone held in right hand | `USING_PHONE` |
| T04 | Phone at left ear | `PHONE_CALL` |
| T05 | Phone at right ear | `PHONE_CALL` |
| T06 | Phone lying on dashboard | `NORMAL` |
| T07 | Phone mounted in holder | Preferably `NORMAL` |
| T08 | Driver drinking from bottle | `NORMAL` |
| T09 | Driver touches face | `NORMAL` |
| T10 | Driver uses radio/control | `NORMAL` |
| T11 | Partial phone occlusion | Evaluate |
| T12 | Low lighting | Evaluate |

For each video, record duration, frames, phone detections, true/detected/false events, recall, precision, average latency, and processing FPS. Keep these reports with the tested thresholds so tuning remains reproducible.

## Current limitations

- COCO phone detection can still miss small, dark, edge-on, or partially occluded phones even after high-resolution local search.
- A 2D wrist/head distance heuristic cannot prove interaction and may confuse nearby dashboard objects or passenger phones.
- Pose keypoints can fail under occlusion, unusual camera angles, PPE, vibration, and low light.
- The initial normalized thresholds and pose-crop scale are starting values, not calibrated safety limits.
- Full-frame inference plus up to four batched pose crops is substantially slower on CPU; CUDA is recommended for a real-time demo.
- The output event begins when the temporal alert activates, so measured detection delay includes the intentional persistence window.

## Recommended next experiment

Run `debug_phone.py` on known missed-phone frames, then process representative fixed-camera videos covering T01-T12 with saved debug frames. Compare model size, full/crop image size, and ROI scale before tuning hand/head distances and the temporal ratio. Select settings that maximize `USING_PHONE` recall while keeping false alarms per hour acceptable. Only after measuring this enhanced pretrained baseline should a custom phone-detector fine-tune be considered.
