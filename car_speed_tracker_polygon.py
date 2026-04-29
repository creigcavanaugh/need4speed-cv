import depthai as dai
import cv2
import time
import numpy as np
from datetime import datetime
import csv
import json
import os

# Polygon-based speed tracker. Uses a 4-point quadrilateral ROI with a perspective
# homography mapping pixel coordinates to real-world feet, so speeds are accurate
# even when the road runs diagonally with depth.
#
# Run define_polygon_roi.py first to populate ROI_POLYGON / ROI_LENGTH_FEET /
# ROI_WIDTH_FEET in config.json.

# --- SETTINGS (defaults; overridden by config.json) ---
ROI_POLYGON = None
ROI_LENGTH_FEET = 60.0
ROI_WIDTH_FEET = 20.0
MIN_BLOB_AREA = 80
FPS = 30
MIN_POINTS_FOR_DISPLAY = 4
MIN_DURATION_SEC = 1.0       # Ignore tracks shorter than this (filters spurious blobs)
MATCH_DIST_FT = 4.0          # Max real-world distance (feet) between consecutive detections to match same track
DIRECTION_TOLERANCE_FT = 1.0 # Per-step backward jitter tolerated; overall start->end trend must dominate
CAMERA_RETRY_DELAY_SEC = 5   # Seconds to wait before reconnecting after a camera error
LOG_FILE = "car_log_polygon.csv"
CAPTURE_IMAGES = False       # Save full-frame JPEG for each logged car
CAPTURE_DIR = "captures"     # Output folder for captured images
DISPLAY_PREVIEW = True
DEBUG = False

_cfg_path = "config.json"
_cfg = {}
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        _cfg = json.load(_f)
    if "ROI_POLYGON" in _cfg:
        ROI_POLYGON = [tuple(p) for p in _cfg["ROI_POLYGON"]]
        print(f"[config] ROI_POLYGON = {ROI_POLYGON}")
    if "ROI_LENGTH_FEET" in _cfg:
        ROI_LENGTH_FEET = float(_cfg["ROI_LENGTH_FEET"])
        print(f"[config] ROI_LENGTH_FEET = {ROI_LENGTH_FEET}")
    if "ROI_WIDTH_FEET" in _cfg:
        ROI_WIDTH_FEET = float(_cfg["ROI_WIDTH_FEET"])
        print(f"[config] ROI_WIDTH_FEET = {ROI_WIDTH_FEET}")
    if "FPS" in _cfg:
        FPS = int(round(_cfg["FPS"]))
        print(f"[config] FPS = {FPS}")
    if "display" in _cfg:
        DISPLAY_PREVIEW = bool(_cfg["display"])
        print(f"[config] display = {DISPLAY_PREVIEW}")
    if "debug" in _cfg:
        DEBUG = bool(_cfg["debug"])
        print(f"[config] debug = {DEBUG}")
    if "min_duration_sec" in _cfg:
        MIN_DURATION_SEC = float(_cfg["min_duration_sec"])
        print(f"[config] min_duration_sec = {MIN_DURATION_SEC}")
    if "capture_images" in _cfg:
        CAPTURE_IMAGES = bool(_cfg["capture_images"])
        print(f"[config] capture_images = {CAPTURE_IMAGES}")
    if "capture_dir" in _cfg:
        CAPTURE_DIR = str(_cfg["capture_dir"])
        print(f"[config] capture_dir = {CAPTURE_DIR}")
    if "match_dist_ft" in _cfg:
        MATCH_DIST_FT = float(_cfg["match_dist_ft"])
        print(f"[config] match_dist_ft = {MATCH_DIST_FT}")
    if "direction_tolerance_ft" in _cfg:
        DIRECTION_TOLERANCE_FT = float(_cfg["direction_tolerance_ft"])
        print(f"[config] direction_tolerance_ft = {DIRECTION_TOLERANCE_FT}")
    if "camera_retry_delay_sec" in _cfg:
        CAMERA_RETRY_DELAY_SEC = float(_cfg["camera_retry_delay_sec"])
        print(f"[config] camera_retry_delay_sec = {CAMERA_RETRY_DELAY_SEC}")

if ROI_POLYGON is None or len(ROI_POLYGON) != 4:
    raise SystemExit("[ERROR] No 4-point ROI_POLYGON in config.json. Run define_polygon_roi.py first.")

if CAPTURE_IMAGES:
    os.makedirs(CAPTURE_DIR, exist_ok=True)


def save_capture(frame, timestamp, car_id, direction, speed_mph):
    if frame is None:
        return
    safe_ts = timestamp.replace(":", "-").replace(" ", "_")
    fname = f"{safe_ts}_car{car_id}_{speed_mph:.0f}mph_{direction}.jpg"
    path = os.path.join(CAPTURE_DIR, fname)
    try:
        cv2.imwrite(path, frame)
        if DEBUG:
            print(f"[DEBUG] saved capture {path}")
    except Exception as e:
        print(f"[WARNING] Could not save capture {path}: {e}")

polygon_np = np.array(ROI_POLYGON, dtype=np.int32)

# Homography: pixel polygon -> real-world rectangle in feet.
# Destination order matches click order: near-left=(0,0), near-right=(W,0),
# far-right=(W,L), far-left=(0,L). So real-world Y increases away from camera.
src_pts = np.array(ROI_POLYGON, dtype=np.float32)
dst_pts = np.array([
    [0.0, 0.0],
    [ROI_WIDTH_FEET, 0.0],
    [ROI_WIDTH_FEET, ROI_LENGTH_FEET],
    [0.0, ROI_LENGTH_FEET],
], dtype=np.float32)
H = cv2.getPerspectiveTransform(src_pts, dst_pts)


def pixel_to_feet(px, py):
    pt = np.array([[[px, py]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


bx, by, bw, bh = cv2.boundingRect(polygon_np)


if LOG_FILE and (not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "car_id", "direction", "speed_mph",
            "speed_ft_s", "num_points", "duration_sec",
            "entry_y_ft", "exit_y_ft"
        ])


def log_car(timestamp, car_id, direction, speed_mph, speed_ft_s,
            num_points, duration_sec, entry_y, exit_y):
    msg = (f"[{timestamp}] Car ID: {car_id} | Points: {num_points} | "
           f"Direction: {direction} | Speed: {speed_mph:.1f} mph "
           f"({speed_ft_s:.1f} ft/s) | Duration: {duration_sec:.2f}s")
    print(msg)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, car_id, direction, f"{speed_mph:.2f}",
                    f"{speed_ft_s:.2f}", num_points, f"{duration_sec:.3f}",
                    f"{entry_y:.2f}", f"{exit_y:.2f}"
                ])
        except IOError as e:
            print(f"[WARNING] Could not write to log file: {e}")


def euclidean_ft(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def direction_check(positions, tol_ft):
    """Return ('AWAY'|'TOWARD', None) if motion is consistent enough, else (None, reason).

    Consistency: overall start->end Y travel must exceed tolerance, and no single
    backward step (against overall direction) may exceed tol_ft.
    """
    if len(positions) < 2:
        return None, "len<2"
    y_first = positions[0][4]
    y_last = positions[-1][4]
    overall = y_last - y_first
    if abs(overall) < tol_ft:
        return None, f"overall_dy={overall:.2f}ft < tol={tol_ft}ft"
    sign = 1.0 if overall > 0 else -1.0
    for i in range(len(positions) - 1):
        step = (positions[i+1][4] - positions[i][4]) * sign
        if step < -tol_ft:
            return None, f"backward_step={step:.2f}ft at i={i}"
    return ("AWAY" if overall > 0 else "TOWARD"), None


def compute_speed_ft(positions):
    if len(positions) < 2:
        return 0.0, 0.0, 0.0
    fx0, fy0 = positions[0][3], positions[0][4]
    fxN, fyN = positions[-1][3], positions[-1][4]
    dist_ft = np.sqrt((fxN - fx0) ** 2 + (fyN - fy0) ** 2)
    dt = positions[-1][2] - positions[0][2]
    if dt == 0:
        return 0.0, 0.0, 0.0
    ft_per_sec = dist_ft / dt
    mph = ft_per_sec * 3600 / 5280
    return ft_per_sec, mph, dt


def flush_trackers(trackers):
    """Log any qualifying tracks still in memory (called on clean exit or before camera restart)."""
    for oid, data in trackers.items():
        positions = data["positions"]
        if data["logged"] or len(positions) < MIN_POINTS_FOR_DISPLAY:
            continue
        direction, _ = direction_check(positions, DIRECTION_TOLERANCE_FT)
        if direction is None:
            continue
        speed_ft, speed_mph, duration = compute_speed_ft(positions)
        if duration < MIN_DURATION_SEC:
            continue
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_car(timestamp, oid, direction, speed_mph, speed_ft,
                len(positions), duration,
                positions[0][4], positions[-1][4])
        if CAPTURE_IMAGES:
            save_capture(data.get("frame"), timestamp, oid, direction, speed_mph)


# mask_full is derived from the polygon and never changes across restarts
mask_full = None
next_id = 0
quit_requested = False

while not quit_requested:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    trackers = {}

    try:
        with dai.Pipeline(dai.Device()) as pipeline:
            pipeline.setXLinkChunkSize(0)
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cap = dai.ImgFrameCapability()
            cap.size.fixed((1920, 1080))
            cap.fps.fixed(FPS)
            xout = cam.requestOutput(cap, True)
            q_video = xout.createOutputQueue(maxSize=4, blocking=False)
            pipeline.start()

            device = pipeline.getDefaultDevice()
            usb_speed = device.getUsbSpeed()
            print(f"[INFO] USB speed: {usb_speed.name}")
            if usb_speed.value < dai.UsbSpeed.SUPER.value:
                print("[WARNING] Not on USB 3 SuperSpeed - frame rate will be limited.")

            if DISPLAY_PREVIEW:
                print("Press 'q' to quit.")
            else:
                print("Running headless. Press Ctrl+C to stop.")

            try:
                while True:
                    img_data = q_video.get()
                    frame = img_data.getCvFrame()

                    if mask_full is None:
                        mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask_full, [polygon_np], 255)

                    frame_crop = frame[by:by+bh, bx:bx+bw]
                    mask_crop = mask_full[by:by+bh, bx:bx+bw]
                    roi_img = cv2.bitwise_and(frame_crop, frame_crop, mask=mask_crop)

                    fg_mask = bg_subtractor.apply(roi_img)
                    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask_crop)

                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    current_frame_time = time.time()

                    detected = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < MIN_BLOB_AREA:
                            if DEBUG:
                                print(f"[DEBUG] filtered area={area:.1f} (MIN_BLOB_AREA={MIN_BLOB_AREA})")
                            continue
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx_local = int(M["m10"] / M["m00"])
                            cy_local = int(M["m01"] / M["m00"])
                            cx_full = cx_local + bx
                            cy_full = cy_local + by
                            fx, fy = pixel_to_feet(cx_full, cy_full)
                            detected.append((cx_full, cy_full, fx, fy))

                    if DEBUG and detected:
                        print(f"[DEBUG] {len(detected)} centroid(s)")

                    updated_ids = set()
                    for cx, cy, fx, fy in detected:
                        min_dist = float('inf')
                        matched_id = None
                        for obj_id, data in trackers.items():
                            if obj_id in updated_ids:
                                continue
                            last = data["positions"][-1]
                            d = euclidean_ft((fx, fy), (last[3], last[4]))
                            if d < MATCH_DIST_FT and d < min_dist:
                                min_dist = d
                                matched_id = obj_id
                        if matched_id is not None:
                            trackers[matched_id]["positions"].append((cx, cy, current_frame_time, fx, fy))
                            if CAPTURE_IMAGES:
                                trackers[matched_id]["frame"] = frame.copy()
                            updated_ids.add(matched_id)
                        else:
                            trackers[next_id] = {
                                "positions": [(cx, cy, current_frame_time, fx, fy)],
                                "logged": False,
                                "frame": frame.copy() if CAPTURE_IMAGES else None,
                            }
                            updated_ids.add(next_id)
                            next_id += 1

                    stale_ids = [oid for oid in trackers if oid not in updated_ids]
                    for oid in stale_ids:
                        data = trackers[oid]
                        positions = data["positions"]
                        if data["logged"]:
                            del trackers[oid]
                            continue
                        if len(positions) < MIN_POINTS_FOR_DISPLAY:
                            if DEBUG:
                                print(f"[DEBUG] dropped track #{oid}: points={len(positions)} < {MIN_POINTS_FOR_DISPLAY}")
                            del trackers[oid]
                            continue
                        direction, dir_reason = direction_check(positions, DIRECTION_TOLERANCE_FT)
                        if direction is None:
                            if DEBUG:
                                print(f"[DEBUG] dropped track #{oid}: direction reject ({dir_reason})")
                            del trackers[oid]
                            continue
                        speed_ft, speed_mph, duration = compute_speed_ft(positions)
                        if duration < MIN_DURATION_SEC:
                            if DEBUG:
                                print(f"[DEBUG] dropped track #{oid}: duration={duration:.2f}s < {MIN_DURATION_SEC}s")
                            del trackers[oid]
                            continue
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_car(timestamp, oid, direction, speed_mph, speed_ft,
                                len(positions), duration,
                                positions[0][4], positions[-1][4])
                        if CAPTURE_IMAGES:
                            save_capture(data.get("frame"), timestamp, oid, direction, speed_mph)
                        del trackers[oid]

                    if DISPLAY_PREVIEW:
                        disp = frame.copy()
                        cv2.polylines(disp, [polygon_np], isClosed=True, color=(0, 0, 255), thickness=2)
                        for obj_id, data in trackers.items():
                            cx, cy = data["positions"][-1][0], data["positions"][-1][1]
                            cv2.circle(disp, (cx, cy), 6, (0, 255, 0), -1)
                            cv2.putText(disp, f"#{obj_id}", (cx + 8, cy - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        cv2.imshow("Detection Mask", fg_mask)
                        cv2.imshow("Speed Tracker (Polygon)", disp)

                        if cv2.waitKey(1) == ord('q'):
                            quit_requested = True
                            break

            except KeyboardInterrupt:
                quit_requested = True

            flush_trackers(trackers)

    except Exception as e:
        flush_trackers(trackers)
        print(f"[ERROR] Camera error: {e}")
        if not quit_requested:
            print(f"[INFO] Restarting in {CAMERA_RETRY_DELAY_SEC:.0f}s... (Ctrl+C to quit)")
            try:
                time.sleep(CAMERA_RETRY_DELAY_SEC)
            except KeyboardInterrupt:
                quit_requested = True

if DISPLAY_PREVIEW:
    cv2.destroyAllWindows()
