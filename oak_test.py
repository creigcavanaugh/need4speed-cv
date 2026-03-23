import depthai as dai
import cv2
import json
import os

# Load display setting from config.json
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DISPLAY_PREVIEW = True
if os.path.exists(_cfg_path):
    with open(_cfg_path) as f:
        _cfg = json.load(f)
    if "display" in _cfg:
        DISPLAY_PREVIEW = bool(_cfg["display"])

print("Connecting to OAK-D...")

with dai.Pipeline(dai.Device()) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cap = dai.ImgFrameCapability()
    cap.size.fixed((1920, 1080))
    cap.fps.fixed(30)
    xout = cam.requestOutput(cap, True)
    q = xout.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    if DISPLAY_PREVIEW:
        print("Connected. Streaming — press 'q' to quit.")
    else:
        print("Connected. Receiving frames (Ctrl+C to stop)...")

    frame_count = 0
    try:
        while True:
            pkt = q.tryGet()
            if pkt is not None:
                frame_count += 1
                frame = pkt.getCvFrame()
                if DISPLAY_PREVIEW:
                    cv2.imshow("OAK-D Stream (Press 'q' to quit)", frame)
                else:
                    print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")

            if DISPLAY_PREVIEW:
                if cv2.waitKey(1) == ord('q'):
                    break
    except KeyboardInterrupt:
        pass

if DISPLAY_PREVIEW:
    cv2.destroyAllWindows()

print(f"Done. Received {frame_count} frames.")
