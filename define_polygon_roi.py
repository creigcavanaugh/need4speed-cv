import depthai as dai
import cv2
import json
import os

# Click 4 corners of the road segment, in order:
#   1: near-left, 2: near-right, 3: far-right, 4: far-left
# This order defines the perspective transform used by car_speed_tracker_polygon.py.

NUM_POINTS = 4
LABELS = ["1: near-left", "2: near-right", "3: far-right", "4: far-left"]
points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < NUM_POINTS:
        points.append((x, y))
        print(f"[INFO] Point {len(points)} ({LABELS[len(points)-1]}): ({x}, {y})")


with dai.Pipeline(dai.Device()) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cap = dai.ImgFrameCapability()
    cap.size.fixed((1920, 1080))
    cap.fps.fixed(30)
    xout = cam.requestOutput(cap, True)
    q = xout.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    window_name = "Define Polygon ROI"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("Click 4 points in order:")
    for lbl in LABELS:
        print(f"  {lbl}")
    print("Keys: 'r' = reset, 's' = save (after 4 points), 'q' = quit")

    saved = False
    while True:
        in_rgb = q.tryGet()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

            for i, p in enumerate(points):
                cv2.circle(frame, p, 6, (0, 0, 255), -1)
                cv2.putText(frame, str(i+1), (p[0]+8, p[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(points) >= 2:
                for i in range(len(points)-1):
                    cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2)
            if len(points) == NUM_POINTS:
                cv2.line(frame, points[-1], points[0], (0, 255, 0), 2)
                cv2.putText(frame, "Polygon complete - press 's' to save",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                hint = LABELS[len(points)]
                cv2.putText(frame, f"Click {hint}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            print("[INFO] Reset.")
        elif key == ord('s') and len(points) == NUM_POINTS:
            cv2.destroyAllWindows()
            try:
                length_ft = float(input("Real-world LENGTH (along travel direction, near->far) in feet: ").strip())
                width_ft = float(input("Real-world WIDTH (across road) in feet: ").strip())
            except ValueError:
                print("[ERROR] Invalid number. Aborting save.")
                break

            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            cfg = {}
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
            cfg["ROI_POLYGON"] = [list(p) for p in points]
            cfg["ROI_LENGTH_FEET"] = length_ft
            cfg["ROI_WIDTH_FEET"] = width_ft
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print("\n" + "="*40)
            print(f"Saved polygon: {points}")
            print(f"  length = {length_ft} ft")
            print(f"  width  = {width_ft} ft")
            print(f"  -> {cfg_path}")
            print("="*40)
            saved = True
            break

cv2.destroyAllWindows()
if not saved:
    print("[INFO] Exited without saving.")
