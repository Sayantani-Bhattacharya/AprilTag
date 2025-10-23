import cv2
import pupil_apriltags as apriltag
import csv
from scipy.spatial.transform import Rotation as R
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pupil_apriltags import Detector
import numpy as np
import os

# ----------------- CONFIG -----------------
bag_path = "rosbag2_2025_10_16-23_29_37"            # ROS2 bag folder
image_topic_name = "/camera/camera/color/image_raw"  # Image topic
fx, fy, cx, cy = 600, 600, 320, 240      # Camera intrinsics
tag_size = 0.16                           # AprilTag size (meters)
output_frames_folder = "frames"           # Optional folder for frames
output_video_path = "rosbag_output.mp4"          # Final video
os.makedirs(output_frames_folder, exist_ok=True)
# ------------------------------------------

# AprilTag detector
# options = apriltag.DetectorOptions(families='tag36h11')
detector = Detector(families='tag36h11')

# ROS2 bag reader
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
converter_options = rosbag2_py.ConverterOptions('', '')
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

# Check if topic exists
topic_types = reader.get_all_topics_and_types()
if image_topic_name not in [t.name for t in topic_types]:
    raise RuntimeError(f"Image topic '{image_topic_name}' not found in bag.")

bridge = CvBridge()
tag_paths = []   # List of dicts: {tag_id, pose, frame_id, timestamp}

frames = []
frame_idx = 0
video_writer = None

try:
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != image_topic_name:
            continue

        # ROS2 image -> OpenCV
        img_msg = deserialize_message(data, Image)
        frame = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        timestamp = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec*1e-9

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=tag_size
        )

        for r in results:
            tag_id = r.tag_id
            tvec = r.pose_t.flatten()        # translation vector (x, y, z)
            R_mat = r.pose_R                 # rotation matrix


            # Regularize/ Orthogonalize Rotation Matrix.
            U, _, Vt = np.linalg.svd(R_mat)
            R_mat = U @ Vt
            if np.linalg.det(R_mat) < 0:
                R_mat *= -1
            quat = R.from_matrix(R_mat).as_quat()

            # quat = R.from_matrix(R_mat).as_quat()  # x, y, z, w

            tag_paths.append({
                "tag_id": tag_id,
                "pose": (tvec[0], tvec[1], tvec[2], quat[0], quat[1], quat[2], quat[3]),
                "frame_id": frame_idx,
                "timestamp": timestamp
            })

            # Draw tag
            pts = [tuple(map(int, corner)) for corner in r.corners]
            cX, cY = map(int, r.center)
            for i in range(4):
                cv2.line(frame, pts[i], pts[(i+1)%4], (0,255,0), 2)
            cv2.circle(frame, (cX, cY), 5, (0,0,255), -1)

        # Initialize video writer once we know frame size
        if video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w,h))

        video_writer.write(frame)
        frames.append(frame)  # optional, in case you want frames saved separately
        frame_idx += 1

finally:
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # Save CSV
    with open('tag_paths.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag_id', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'frame_id', 'timestamp'])
        for entry in tag_paths:
            writer.writerow([
                entry['tag_id'],
                *entry['pose'],
                entry['frame_id'],
                entry['timestamp']
            ])

    print(f"Processed {frame_idx} frames. Video saved to '{output_video_path}' and CSV saved to 'tag_paths.csv'.")
