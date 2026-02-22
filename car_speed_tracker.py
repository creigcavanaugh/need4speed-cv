import depthai as dai
import cv2
import time
import numpy as np
from datetime import datetime
import csv
import os

# Car Speed Estimation using DepthAI and OpenCV

# --- SETTINGS ---
ROI_LANE = (771, 355, 98, 50)
MIN_BLOB_AREA = 80
FPS = 30
MIN_POINTS_FOR_DISPLAY = 4
ROI_X_FEET = 58.0           # Real-world width of ROI in feet
LOG_FILE = "car_log.csv"     # CSV log file path, or "" to disable

# --- PIPELINE SETUP ---
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setFps(FPS)
cam.setInterleaved(False)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.video.link(xout.input)

# Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

# Tracking dictionary
trackers = {}
next_id = 0

# Precompute pixels-to-feet ratio
ROI_WIDTH_PX = ROI_LANE[2]
FEET_PER_PIXEL = ROI_X_FEET / ROI_WIDTH_PX if ROI_WIDTH_PX > 0 else 0

# Write CSV header if log file is new or empty
if LOG_FILE and (not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "car_id", "direction", "speed_mph",
            "speed_px_s", "num_points", "duration_sec",
            "entry_x", "exit_x"
        ])


def log_car(timestamp, car_id, direction, speed_mph, speed_px_s,
            num_points, duration_sec, entry_x, exit_x):
    """Print to console and append to CSV log."""
    msg = (f"[{timestamp}] Car ID: {car_id} | Points: {num_points} | "
           f"Direction: {direction} | Speed: {speed_mph:.1f} mph "
           f"({speed_px_s:.1f} px/s) | Duration: {duration_sec:.2f}s")
    print(msg)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, car_id, direction, f"{speed_mph:.2f}",
                    f"{speed_px_s:.2f}", num_points, f"{duration_sec:.3f}",
                    entry_x, exit_x
                ])
        except IOError as e:
            print(f"[WARNING] Could not write to log file: {e}")


def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_consistent_direction(positions):
    if len(positions) < 2:
        return False
    deltas = [positions[i+1][0] - positions[i][0] for i in range(len(positions) - 1)]
    nonzero_deltas = [d for d in deltas if d != 0]
    if not nonzero_deltas:
        return False
    return all(d > 0 for d in nonzero_deltas) or all(d < 0 for d in nonzero_deltas)


def compute_speed(positions):
    if len(positions) < 2:
        return 0.0, 0.0, 0.0
    first = positions[0]
    last = positions[-1]
    dx = abs(last[0] - first[0])
    dt = last[2] - first[2]
    if dt == 0:
        return 0.0, 0.0, 0.0
    px_per_sec = dx / dt
    feet_per_sec = px_per_sec * FEET_PER_PIXEL
    mph = feet_per_sec * 3600 / 5280
    return px_per_sec, mph, dt


with dai.Device(pipeline) as device:
    q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)

    while True:
        img_data = q_video.get()
        frame = img_data.getCvFrame()

        x, y, w, h = ROI_LANE
        roi_img = frame[y:y+h, x:x+w]

        fg_mask = bg_subtractor.apply(roi_img)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_time = time.time()

        detected_centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_BLOB_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected_centroids.append((cx, cy))

        updated_ids = set()
        for centroid in detected_centroids:
            min_dist = float('inf')
            matched_id = None
            for obj_id, data in trackers.items():
                if obj_id in updated_ids:
                    continue
                last_pos = data["positions"][-1][:2]
                dist = euclidean_distance(centroid, last_pos)
                if dist < 40 and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id
            if matched_id is not None:
                trackers[matched_id]["positions"].append((centroid[0], centroid[1], current_frame_time))
                updated_ids.add(matched_id)
            else:
                trackers[next_id] = {
                    "positions": [(centroid[0], centroid[1], current_frame_time)],
                    "logged": False
                }
                updated_ids.add(next_id)
                next_id += 1

        # Log cars that just left the ROI (stale) and qualified
        stale_ids = [oid for oid in trackers if oid not in updated_ids]
        for oid in stale_ids:
            data = trackers[oid]
            positions = data["positions"]
            if (len(positions) >= MIN_POINTS_FOR_DISPLAY
                    and is_consistent_direction(positions)
                    and not data["logged"]):
                speed_px, speed_mph, duration = compute_speed(positions)
                direction = "RIGHT" if positions[-1][0] > positions[0][0] else "LEFT"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_car(timestamp, oid, direction, speed_mph, speed_px,
                        len(positions), duration, positions[0][0], positions[-1][0])
            del trackers[oid]

        # Also log active cars that hit the threshold (for live feedback)
        for obj_id, data in trackers.items():
            positions = data["positions"]
            cx, cy = positions[-1][0], positions[-1][1]
            cv2.circle(roi_img, (cx, cy), 5, (0, 255, 0), -1)
            #cv2.putText(roi_img, f"ID:{obj_id}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detection Mask", fg_mask)
        cv2.imshow("Speed Tracker", roi_img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Log any remaining tracked cars on exit
    for oid, data in trackers.items():
        positions = data["positions"]
        if (len(positions) >= MIN_POINTS_FOR_DISPLAY
                and is_consistent_direction(positions)
                and not data["logged"]):
            speed_px, speed_mph, duration = compute_speed(positions)
            direction = "RIGHT" if positions[-1][0] > positions[0][0] else "LEFT"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_car(timestamp, oid, direction, speed_mph, speed_px,
                    len(positions), duration, positions[0][0], positions[-1][0])

cv2.destroyAllWindows()
