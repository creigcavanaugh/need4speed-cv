import depthai as dai
import cv2
import sys

# --- DEBUG CHECK ---
# If this prints "List of nodes: ..." and includes 'XLinkOut', your install is fine.
# If it fails here, you likely have a local file named 'depthai.py' shadowing the library.
try:
    if not hasattr(dai.node, 'XLinkOut'):
        print("CRITICAL ERROR: 'XLinkOut' not found in depthai.node.")
        print("1. Check if you have a file named 'depthai.py' in your folder.")
        print("2. If not, try running: pip install --upgrade --force-reinstall depthai")
        print("Available nodes:", dir(dai.node))
        sys.exit(1)
except Exception as e:
    print(f"Debug check failed: {e}")
# -------------------

# 1. Create the pipeline
pipeline = dai.Pipeline()

# 2. Define source - Unified "Camera" Node (Fixes Deprecation Warning)
# This node handles both Color and Mono cameras
cam = pipeline.create(dai.node.Camera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A) # CAM_A is usually the RGB center camera
cam.setSize(1920, 1080)                         # Set resolution (1080p)
cam.setFps(30)

# 3. Define output - XLinkOut
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")

# Link the camera 'video' output to the XLink input
cam.video.link(xout_rgb.input)

# 4. Connect to device
print("Connecting to OAK-D...")
with dai.Device(pipeline) as device:
    print("Connected. Starting pipeline...")
    
    # Output queue
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.tryGet()

        if in_rgb is not None:
            # The Camera node output is NV12 by default on some versions, 
            # getCvFrame() handles the conversion automatically.
            frame = in_rgb.getCvFrame()

            cv2.imshow("OAK-D Stream (Press 'q' to quit)", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    