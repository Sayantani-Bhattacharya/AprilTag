# pyenv activate realsense-env

import pyrealsense2 as rs
import numpy as np
import cv2
import apriltag

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config() 
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Set up AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11') # tagStandard41h12
detector = apriltag.Detector(options)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        results = detector.detect(gray)

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = tuple(map(int, ptA))
            ptB = tuple(map(int, ptB))
            ptC = tuple(map(int, ptC))
            ptD = tuple(map(int, ptD))
            cX, cY = map(int, r.center)

            cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)
            cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(color_image, f"ID: {r.tag_id}", (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(1)
        if key == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
