# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time vehicle speed monitoring system using a Luxonis OAK-D camera. Detects vehicles in a defined road segment, estimates their speed in mph, and generates PDF analysis reports.

## Setup

```bash
pip install -r requirements.txt
```

Requires a physical Luxonis OAK-D camera connected via USB.

## Workflow

The system runs in three sequential steps:

```bash
python define_roi.py        # Step 1: Interactively draw ROI on live video, prints (x,y,w,h) to copy into car_speed_tracker.py
python car_speed_tracker.py # Step 2: Run live tracker, logs to car_log.csv
python generate_report.py   # Step 3: Generate PDF report from car_log.csv
```

`oak_test.py` is a diagnostic utility for validating camera connection.

## Configuration

All tunable parameters are at the top of `car_speed_tracker.py`:

- `ROI_LANE` — ROI position `(x, y, w, h)` in pixels (get from `define_roi.py`)
- `ROI_X_FEET` — real-world width of the ROI in feet (must be measured physically)
- `MIN_BLOB_AREA` — minimum contour area in pixels² to count as a vehicle
- `FPS` — camera frame rate
- `MIN_POINTS_FOR_DISPLAY` — minimum tracking points required to log a vehicle pass
- `LOG_FILE` — CSV output path; set to `""` to disable logging
- `DISPLAY_PREVIEW` — show live cv2 windows; set `False` (or `"display": false` in `config.json`) for headless environments

## Architecture

**Speed estimation pipeline** (`car_speed_tracker.py`):
1. DepthAI pipeline streams frames from OAK-D camera
2. MOG2 background subtraction isolates moving objects in the ROI
3. Contour detection → centroid extraction per frame
4. Centroids matched to existing tracks by Euclidean distance (< 40px threshold)
5. When a track goes stale: compute `dx/dt` in pixels/sec → convert to mph using `ROI_X_FEET / ROI_width_px`
6. Filter tracks by unidirectional movement; log direction (LEFT/RIGHT), speed, timestamp to CSV

**Report generation** (`generate_report.py`):
- Parses `car_log.csv`, computes statistics (median, mean, percentiles), groups by day/hour
- Produces 5 chart types via Matplotlib, assembles multi-page PDF via ReportLab
