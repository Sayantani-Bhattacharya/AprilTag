import cv2
import apriltag
import numpy as np

# Set up AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')  
detector = apriltag.Detector(options)

# Open the video file
video_path = "shoreArray2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

try:
    frames = []  # To store frames for saving the final video
    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = tuple(map(int, ptA))
            ptB = tuple(map(int, ptB))
            ptC = tuple(map(int, ptC))
            ptD = tuple(map(int, ptD))
            cX, cY = map(int, r.center)

            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {r.tag_id}", (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("AprilTag Detection", frame)
        frames.append(frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break



finally:
    # Save the processed frames as a video
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    cap.release()
    cv2.destroyAllWindows()
