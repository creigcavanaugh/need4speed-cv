import depthai as dai
import cv2
import json
import os

# Global variables to store mouse interaction state
roi_defined = False
start_point = None
end_point = None
roi_coordinates = (0, 0, 0, 0) # x, y, w, h

def select_roi(event, x, y, flags, param):
    global start_point, end_point, roi_defined, roi_coordinates

    # Left mouse button clicked: start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        roi_defined = False

    # Mouse moving: update rectangle visualization
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        end_point = (x, y)

    # Left mouse button released: finalize region
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        roi_defined = True

        # Calculate x, y, w, h ensuring positive width/height
        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        roi_coordinates = (x1, y1, x2-x1, y2-y1)
        print(f"\n[INFO] ROI Selected: {roi_coordinates}")
        print("Press 's' to save and exit, or draw again.")

with dai.Pipeline(dai.Device()) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cap = dai.ImgFrameCapability()
    cap.size.fixed((1920, 1080))
    cap.fps.fixed(30)
    xout = cam.requestOutput(cap, True)
    q = xout.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    # Create window and attach mouse callback
    window_name = "Draw Box over Road (Press 's' to save)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_roi)

    print("Draw a box over the road segment.")

    while True:
        in_rgb = q.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

            # If user is drawing, show the rectangle
            if start_point and end_point:
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            # If ROI is finalized, keep showing it
            if roi_defined:
                x, y, w, h = roi_coordinates
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "ROI Selected", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s') and roi_defined:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            cfg = {}
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
            cfg["ROI_LANE"] = list(roi_coordinates)
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print("\n" + "="*30)
            print(f"FINAL ROI TO USE: {roi_coordinates}")
            print(f"Saved to {cfg_path}")
            print("="*30)
            break

cv2.destroyAllWindows()
