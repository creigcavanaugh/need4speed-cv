import depthai as dai
import cv2

print("Connecting to OAK-D...")

with dai.Pipeline(dai.Device()) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cap = dai.ImgFrameCapability()
    cap.size.fixed((1920, 1080))
    cap.fps.fixed(30)
    xout = cam.requestOutput(cap, True)
    q = xout.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()

    print("Connected. Streaming — press 'q' to quit.")

    while True:
        pkt = q.tryGet()
        if pkt is not None:
            cv2.imshow("OAK-D Stream (Press 'q' to quit)", pkt.getCvFrame())
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
