import cv2
import apriltag
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R

# Camera intrinsics (fx, fy, cx, cy) and tag size (in meters)

# # External Tag
# fx, fy, cx, cy = 600, 600, 320, 240  
# tag_size = 0.16  # Replace with your actual tag size                                    ---> IMP TO SWITCH 


# Internal Tag
fx, fy, cx, cy = 1806.68775, 1801.14087, 1882.18218, 1404.07528   # With OpenCV Calib Matrix.
tag_size = 0.007725     # The black square width ≈ 0.75 × 10.3 mm = 7.725 mm.   || 10.3 mm is with border


 
# Set up AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Open the video file
video_path = "T01.mp4"  #shoreArray2.mp4 | GlareCheck.mp4  | Last: "T4.mp4" Oct 14th
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

tag_paths = {}  # {tag_id: [(frame_idx, tx, ty, tz, qx, qy, qz, qw), ...]}
frame_idx = 0

try:
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        for r in results:
            tag_id = r.tag_id
            M, init_error, final_error = detector.detection_pose(r, [fx, fy, cx, cy], tag_size)
            t = M[:3, 3]  # translation vector
            R_mat = M[:3, :3]  # rotation matrix
            quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)

            pose = (frame_idx, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
            tag_paths.setdefault(tag_id, []).append(pose)

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
            cv2.putText(frame, f"ID: {tag_id}", (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("AprilTag Detection", frame)
        frames.append(frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break
        frame_idx += 1

finally:
    # Save the processed frames as a video
    if frames:
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
        for f in frames:
            out.write(f)
        out.release()
    cap.release()
    cv2.destroyAllWindows()

    # Save tag paths to CSV
    with open('tag_paths.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tag_id', 'frame_idx', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        for tag_id, poses in tag_paths.items():
            for pose in poses:
                writer.writerow([tag_id] + list(pose))