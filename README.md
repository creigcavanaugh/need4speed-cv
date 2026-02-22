# need4speed-cv

Real-time vehicle speed monitoring using computer vision. Tracks vehicles passing through a defined road segment, estimates their speed in mph, and generates detailed PDF analysis reports with per-day breakdowns.

Built for the [Luxonis OAK-D](https://docs.luxonis.com/hardware/products/OAK-D/) camera using DepthAI and OpenCV.

## Features

- Live vehicle detection and tracking via background subtraction
- Speed estimation in mph using a calibrated real-world ROI width
- Direction tracking (left/rightbound)
- CSV logging with per-vehicle timestamps, speeds, tracking points, and durations
- PDF report generation with:
  - Overall and per-day speed statistics (median, mean, 85th percentile, max)
  - Traffic volume and throughput metrics
  - Speed distribution histograms and cumulative distribution charts
  - Hourly volume bar charts and speed-over-time scatter plots
  - Multi-day comparison charts and tables

## Scripts

| Script | Description |
|---|---|
| `define_roi.py` | Live camera tool to visually draw and save your ROI coordinates |
| `car_speed_tracker.py` | Main tracker — runs live detection, logs results to `car_log.csv` |
| `generate_report.py` | Reads `car_log.csv` and produces a PDF report |

## Setup

### Requirements

- Python 3.8+
- Luxonis OAK-D camera

### Install dependencies

```bash
pip install depthai opencv-python numpy matplotlib reportlab
```

## Usage

### 1. Define your ROI

Run `define_roi.py` with your OAK-D connected. Draw a box over the road segment you want to monitor and press `s` to print the ROI coordinates.

```bash
python define_roi.py
```

Copy the printed `(x, y, w, h)` tuple into `car_speed_tracker.py` as `ROI_LANE`, and set `ROI_X_FEET` to the real-world width of that segment in feet.

### 2. Run the tracker

```bash
python car_speed_tracker.py
```

Press `q` to quit. Speed readings are logged to `car_log.csv`.

### 3. Generate a report

```bash
python generate_report.py
python generate_report.py --input car_log.csv --output report.pdf
```

## Configuration

Edit the settings at the top of `car_speed_tracker.py`:

| Variable | Default | Description |
|---|---|---|
| `ROI_LANE` | `(771, 355, 98, 50)` | ROI position and size in pixels `(x, y, w, h)` |
| `ROI_X_FEET` | `58.0` | Real-world width of the ROI in feet |
| `MIN_BLOB_AREA` | `80` | Minimum contour area (px²) to count as a vehicle |
| `FPS` | `30` | Camera frame rate |
| `MIN_POINTS_FOR_DISPLAY` | `4` | Minimum tracking points required to log a vehicle |
| `LOG_FILE` | `car_log.csv` | CSV output path, or `""` to disable logging |

## License

See [LICENSE](LICENSE).
