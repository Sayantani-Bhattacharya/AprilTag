import cv2
import apriltag
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import sys
import matplotlib.pyplot as plt
from sixDPoseVisualize import plot_6d_pose


# Camera intrinsics (fx, fy, cx, cy) and tag size (in meters)
fx, fy, cx, cy = 600, 600, 320, 240  # Replace with RealSense intrinsics
tag_size = 0.16  # Replace with actual tag size

# Set up AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

tag_paths = {}  # {tag_id: [(frame_idx, tx, ty, tz, qx, qy, qz, qw), ...]}
frame_idx = 0
frames = []

# Create the figure and axes once
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

try:
    while True:
        frameset = pipeline.wait_for_frames()
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Grayscale for AprilTag detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        

        for r in results:
            tag_id = r.tag_id

            # Compute pose using intrinsics and tag size
            M, init_error, final_error = detector.detection_pose(r, [fx, fy, cx, cy], tag_size)
            t = M[:3, 3]  # translation
            R_mat = M[:3, :3]  # rotation
            quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)

            pose = (frame_idx, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
            tag_paths.setdefault(tag_id, []).append(pose)

            # Draw tag outline
            ptA, ptB, ptC, ptD = map(tuple, r.corners)
            ptA, ptB, ptC, ptD = tuple(map(int, ptA)), tuple(map(int, ptB)), tuple(map(int, ptC)), tuple(map(int, ptD))
            cX, cY = map(int, r.center)

            cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)
            cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(color_image, f"ID: {tag_id}", (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Visual: combine color + depth for display
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        if depth_colormap.shape != color_image.shape:
            color_resized = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
            display_image = np.hstack((color_resized, depth_colormap))
        else:
            display_image = np.hstack((color_image, depth_colormap))

        cv2.imshow("RealSense + AprilTags", display_image)
        frames.append(display_image)

        # Save tag paths to csv every 100 frames and save video at the end
        if frame_idx % 10 == 0 and frame_idx > 0:
            with open('tag_paths.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['tag_id', 'frame_idx', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
                for tag_id, poses in tag_paths.items():
                    for pose in poses:
                        writer.writerow([tag_id] + list(pose))
            
            
            plot_6d_pose(ax)


            

            # # Call the plot function dynamically every 100 frames
            # if frame_idx % 100 == 0:
            #     plot_6d_pose(ax)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        frame_idx += 1

finally:
    # Save video
    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
        for f in frames:
            out.write(f)
        out.release()

    # # Save tag paths to CSV
    # with open('tag_paths.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['tag_id', 'frame_idx', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    #     for tag_id, poses in tag_paths.items():
    #         for pose in poses:
    #             writer.writerow([tag_id] + list(pose))

    pipeline.stop()
    cv2.destroyAllWindows()







# To do: 
# 1. dynamic plotting of 6d pose path every 100 frames: jyooti sent check that
# 2. saving the static tag and calculate relative pose of dynamic tags to the static tag.
# 3. transforming id based locations.

# Kal give stuff for printing, and then complete this: and record a video showing pura rotate karne pe pose is same, and translate karne pe reflectted same as it physically moves.

# then start with wisker movements.: fan stuff:   do it on fri and saturday!!