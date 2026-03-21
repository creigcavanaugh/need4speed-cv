#!/usr/bin/env python3
"""
auto_roi.py — Automatically detect the active road ROI using a motion heatmap.

Modes:
  interactive  Show live heatmap + detected ROI rectangle.
               Press 's' to patch ROI_LANE in car_speed_tracker.py and exit.
               Press 'q' to quit without saving.

  auto         Run headless. Watch for --duration seconds OR until --confidence
               is reached (whichever comes first), then patch ROI_LANE in
               car_speed_tracker.py and launch it automatically.

Usage:
  python auto_roi.py                                  # interactive (default)
  python auto_roi.py --mode interactive
  python auto_roi.py --mode auto --duration 30
  python auto_roi.py --mode auto --confidence 0.85
  python auto_roi.py --mode auto --duration 60 --confidence 0.90
"""

import argparse
import re
import subprocess
import sys
import time
from collections import deque

import cv2
import depthai as dai
import numpy as np

# --- Tunable constants ---
TRACKER_SCRIPT   = "car_speed_tracker.py"
FPS              = 30
DECAY            = 0.97   # Exponential decay for heatmap accumulator (higher = longer memory)
HEATMAP_THRESH   = 55     # 0-255 threshold applied to normalized heatmap
MORPH_KERNEL     = 20     # Morphological close kernel size (pixels) to fill gaps
MIN_REGION_AREA  = 1000   # Min thresholded-heatmap area (px²) for a region to be valid
STABILITY_WINDOW = 90     # Rolling frame window used to compute confidence
STABILITY_TOL    = 8      # Max pixel range in any bbox dimension to be "stable"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(FPS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.video.link(xout.input)
    return pipeline


# ---------------------------------------------------------------------------
# ROI detection helpers
# ---------------------------------------------------------------------------

def detect_roi(heatmap_norm):
    """
    Return (x, y, w, h) bounding rect of the largest contiguous active region
    in the normalised heatmap, or None if nothing qualifies.
    """
    _, thresh = cv2.threshold(heatmap_norm, HEATMAP_THRESH, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_REGION_AREA:
        return None
    return cv2.boundingRect(largest)


def compute_confidence(history: deque) -> float:
    """
    Return a confidence score [0, 1] based on how stable the recent bounding
    rects have been.  Requires at least half a full window before scoring.
    """
    if len(history) < STABILITY_WINDOW // 2:
        return 0.0
    arr = np.array(history, dtype=float)           # shape (N, 4)
    spread = arr.max(axis=0) - arr.min(axis=0)     # range per dimension
    max_spread = float(spread.max())
    # Linear ramp: 0 spread → 1.0, STABILITY_TOL*5 spread → 0.0
    return max(0.0, min(1.0, 1.0 - max_spread / (STABILITY_TOL * 5)))


# ---------------------------------------------------------------------------
# Patching car_speed_tracker.py
# ---------------------------------------------------------------------------

def patch_tracker(roi: tuple) -> bool:
    """Replace ROI_LANE in car_speed_tracker.py with the detected roi."""
    x, y, w, h = roi
    try:
        with open(TRACKER_SCRIPT, "r") as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[ERROR] {TRACKER_SCRIPT} not found.")
        return False

    new_src = re.sub(
        r"ROI_LANE\s*=\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)",
        f"ROI_LANE = ({x}, {y}, {w}, {h})",
        src,
    )
    if new_src == src:
        print(f"[WARNING] ROI_LANE pattern not found in {TRACKER_SCRIPT} — no changes made.")
        return False

    with open(TRACKER_SCRIPT, "w") as f:
        f.write(new_src)
    print(f"[INFO] Patched {TRACKER_SCRIPT}  →  ROI_LANE = ({x}, {y}, {w}, {h})")
    return True


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive():
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=False
    )
    pipeline = build_pipeline()
    accumulator = None
    history: deque = deque(maxlen=STABILITY_WINDOW)
    roi_rect = None

    print("Watching for motion...")
    print("  's'  — save detected ROI to car_speed_tracker.py and exit")
    print("  'q'  — quit without saving")

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        win = "Auto ROI  |  's' save  'q' quit"
        cv2.namedWindow(win)

        while True:
            in_frame = q.tryGet()
            if in_frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Quit without saving.")
                    break
                continue

            frame = in_frame.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = bg_sub.apply(gray)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

            fg_f = fg.astype(np.float32) / 255.0
            accumulator = fg_f if accumulator is None else accumulator * DECAY + fg_f * (1 - DECAY)

            heatmap_norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            display = cv2.addWeighted(frame, 0.5, heatmap_color, 0.5, 0)

            roi_rect = detect_roi(heatmap_norm)
            if roi_rect:
                history.append(roi_rect)
                x, y, w, h = roi_rect
                conf = compute_confidence(history)
                color = (0, 255, 0) if conf >= 0.7 else (0, 200, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                label = f"ROI ({x},{y},{w},{h})  conf {conf:.0%}"
                cv2.putText(display, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                cv2.putText(display, "Waiting for motion...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit without saving.")
                break
            elif key == ord("s"):
                if roi_rect:
                    patch_tracker(roi_rect)
                    break
                else:
                    print("[WARNING] No ROI detected yet — keep watching.")

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Auto mode
# ---------------------------------------------------------------------------

def run_auto(duration_sec: float, confidence_threshold: float):
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=False
    )
    pipeline = build_pipeline()
    accumulator = None
    history: deque = deque(maxlen=STABILITY_WINDOW)
    best_roi = None
    deadline = time.time() + duration_sec

    print(f"[AUTO] Observing for up to {duration_sec:.0f}s "
          f"or until {confidence_threshold:.0%} confidence...")

    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_frame = q.tryGet()
            if in_frame is None:
                time.sleep(0.01)
                # Still respect deadline even when no frames arrive
                if time.time() >= deadline:
                    break
                continue

            frame = in_frame.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = bg_sub.apply(gray)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

            fg_f = fg.astype(np.float32) / 255.0
            accumulator = fg_f if accumulator is None else accumulator * DECAY + fg_f * (1 - DECAY)

            heatmap_norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            roi_rect = detect_roi(heatmap_norm)

            if roi_rect:
                history.append(roi_rect)
                best_roi = roi_rect
                conf = compute_confidence(history)
                elapsed = time.time() - (deadline - duration_sec)
                remaining = max(0.0, deadline - time.time())
                print(
                    f"\r[AUTO] {elapsed:5.1f}s  ROI={roi_rect}  conf={conf:.0%}  "
                    f"(deadline in {remaining:.0f}s)   ",
                    end="",
                    flush=True,
                )
                if conf >= confidence_threshold:
                    print(f"\n[AUTO] Confidence threshold {confidence_threshold:.0%} reached.")
                    break
            else:
                elapsed = time.time() - (deadline - duration_sec)
                print(f"\r[AUTO] {elapsed:5.1f}s  Waiting for motion...                    ",
                      end="", flush=True)

            if time.time() >= deadline:
                print(f"\n[AUTO] Observation window ({duration_sec:.0f}s) elapsed.")
                break

    if best_roi is None:
        print("[ERROR] No ROI detected during observation window. Exiting.")
        sys.exit(1)

    print(f"[AUTO] Using ROI: {best_roi}")
    if not patch_tracker(best_roi):
        sys.exit(1)

    print(f"[AUTO] Launching {TRACKER_SCRIPT} ...")
    subprocess.run([sys.executable, TRACKER_SCRIPT], check=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect active road ROI from motion and optionally launch the speed tracker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["interactive", "auto"], default="interactive",
        help="interactive: live heatmap, confirm with 's'; "
             "auto: headless, patch ROI and launch tracker (default: interactive)",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0, metavar="SECONDS",
        help="(auto) Max seconds to observe before committing to best ROI (default: 30)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.80, metavar="0-1",
        help="(auto) Stop early when ROI stability confidence reaches this value (default: 0.80)",
    )
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive()
    else:
        if not (0.0 < args.confidence <= 1.0):
            parser.error("--confidence must be between 0 (exclusive) and 1 (inclusive)")
        if args.duration <= 0:
            parser.error("--duration must be positive")
        run_auto(args.duration, args.confidence)


if __name__ == "__main__":
    main()
