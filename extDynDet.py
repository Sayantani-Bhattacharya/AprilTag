import cv2
import apriltag
import numpy as np
import csv
from collections import deque
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import sys
import matplotlib.pyplot as plt
from sixDPoseVisualize import plot_6d_pose, plot_3d_relative_pose


# Camera intrinsics (fx, fy, cx, cy) and tag size (in meters)
fx, fy, cx, cy = 600, 600, 320, 240  # Replace with RealSense intrinsics
tag_size = 0.16  # Replace with actual tag size

# Set up AprilTag detector
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)

# External Cube Dimensions
cube_size = 0.166  # meters   


# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

tag_paths = {}  # {tag_id: [(frame_idx, tx, ty, tz, qx, qy, qz, qw), ...]}
frame_idx = 0
frames = []

fused_centroid_path = [] 


# Creating the figure and axes once for tag path plotting.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Creating the figure and axes once for relative pose plotting.
fig_relative = plt.figure()
ax_relative = fig_relative.add_subplot(111, projection='3d')

# Creating the figure and axes once for cube centroid plotting.
fig_centroid = plt.figure()
ax_centroid = fig_centroid.add_subplot(111, projection='3d')


# Defining rigid transformation matrix from each id to centroid [ Assuming facce 1 as front]
# T_1C: 1 wrt Centroid (0,0,0)
# T_C1: Centroid (0,0,0) wrt 1
T_1C = np.array([[1, 0, 0, cube_size / 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

T_C1 = np.array([[1, 0, 0, -cube_size / 2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

T_2C = np.array([[0, 0, 1, 0],
                [0, 1, 0, cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C2 = np.array([[0, 0, 1, 0],
                [0, 1, 0, -cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_3C = np.array([[0, 0, 1, -cube_size / 2],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C3 = np.array([[0, 0, 1, cube_size / 2],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_4C = np.array([[0, 0, 1, 0],
                [0, 1, 0, -cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_C4 = np.array([[0, 0, 1, 0],
                [0, 1, 0, cube_size / 2],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]])

T_5C = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, cube_size / 2],
                [0, 0, 0, 1]])

T_C5 = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, - cube_size / 2],
                [0, 0, 0, 1]])




# class PoseFilterWindowEMA:
#     """
#     Pose filter combining:
#       - small sliding window for outlier rejection (median)
#       - EMA smoothing for real-time updates
    
#     Filters 6-DoF poses given as (frame_idx, t_x, t_y, t_z, q_x, q_y, q_z, q_w)
#     """

#     def __init__(self, window_size=5, alpha_pos=0.3, alpha_rot=0.3):
#         """
#         window_size: number of past poses to keep for median
#         alpha_pos: EMA factor for translation (0..1)
#         alpha_rot: EMA factor for rotation (0..1)
#         """
#         self.window_size = window_size
#         self.alpha_pos = alpha_pos
#         self.alpha_rot = alpha_rot
#         self.pos_window = deque(maxlen=window_size)
#         self.rot_window = deque(maxlen=window_size)  # store Rotation objects
#         # EMA internal state
#         self.prev_pos = None
#         self.prev_rot = None

#     def reset(self):
#         self.pos_window.clear()
#         self.rot_window.clear()
#         self.prev_pos = None
#         self.prev_rot = None

#     def filter(self, pos_new, rotmat_new):
#         """
#         pos_new: np.array (3,)
#         rotmat_new: np.array (3,3)
#         Returns filtered (pos, rotmat)
#         """
#         # --- add new measurement to window ---
#         self.pos_window.append(pos_new)
#         self.rot_window.append(R.from_matrix(rotmat_new))

#         # --- median of positions ---
#         pos_stack = np.stack(self.pos_window)
#         pos_med = np.median(pos_stack, axis=0)

#         # --- median-like rotation (approx): pick rotation closest to median position ---
#         # Simple approach: pick rotation in window closest to median translation
#         dists = np.linalg.norm(pos_stack - pos_med, axis=1)
#         rot_med = self.rot_window[np.argmin(dists)]

#         # --- EMA smoothing ---
#         if self.prev_pos is None:
#             self.prev_pos = pos_med
#             self.prev_rot = rot_med
#             return self.prev_pos, self.prev_rot.as_matrix()

#         # EMA for position
#         self.prev_pos = (1 - self.alpha_pos) * self.prev_pos + self.alpha_pos * pos_med

#         # EMA for rotation: SLERP
#         slerp = R.slerp(0, 1, [self.prev_rot, rot_med])
#         self.prev_rot = slerp([self.alpha_rot])[0]

#         return self.prev_pos, self.prev_rot.as_matrix()



def plot_centroid(ax_centroid, t, tag_id):
    # t: transform vector of the tag wrt camera.
    # T_centroid: transform matrix of the centroid wrt camera.  

    if tag_id == 1:
        T_centroid =  T_C1 * t 
    elif tag_id == 2:
        T_centroid = T_C2 * t
    elif tag_id == 3:
        T_centroid =  T_C3 * t
    elif tag_id == 4:
        T_centroid =  T_C4 * t
    elif tag_id == 5:
        T_centroid =  T_C5 * t
    else:
        # should be an impossible case
        T_centroid =  np.eye(4)

    # Extract translation
    centroid = T_centroid[:3, 3]

    # Plot centroid
    # ax_centroid.clear()
    ax_centroid.scatter(centroid[0], centroid[1], centroid[2], c='r', marker='o') # s=5
    ax_centroid.set_xlabel('X (m)')
    ax_centroid.set_ylabel('Y (m)')
    ax_centroid.set_zlabel('Z (m)')
    # ax_centroid.legend(['Centroid'])
    ax_centroid.set_title('Cube Centroid Path')
    plt.show(block=False)
    plt.pause(0.01)  # Pause to update the plot

def plot_centroid_path(ax_centroid, current_frame_poses):   
    global fused_centroid_path
    # Clear the plot
    ax_centroid.clear()

    # Centroid poses from all tags
    centroid_poses = []

    # Iterate through all tag paths and plot their centroid paths
    for pose in current_frame_poses:
        centroids = []
        # for pose in poses:
        tag_id = pose[0]
        _, tx, ty, tz, qx, qy, qz, qw = pose
        t = np.array([[1, 0, 0, tx],
                        [0, 1, 0, ty],
                        [0, 0, 1, tz],
                        [0, 0, 0, 1]])
        if tag_id == 1:
            T_centroid = T_C1 @ t
        elif tag_id == 2:
            T_centroid = T_C2 @ t
        elif tag_id == 3:
            T_centroid = T_C3 @ t
        elif tag_id == 4:
            T_centroid = T_C4 @ t
        elif tag_id == 5:
            T_centroid = T_C5 @ t
        else:
            T_centroid = np.eye(4)

        # Extract translation and orientation
        tx, ty, tz = T_centroid[:3, 3]
        rotation_matrix = T_centroid[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()  # (qx, qy, qz, qw)
        qx, qy, qz, qw = quaternion
        centroid = [tx, ty, tz, qx, qy, qz, qw]
        centroid_poses.append(centroid)
        
    # fuse the centroid poses from all visible tags to get a more stable estimate.
    fused_pose = pose_fusion(centroid_poses)


    # TODO: No visible tags case: currently returning None, should handle it better.
    if fused_pose is not None:
        fused_centroid_path.append(fused_pose)

    # print("Fused Pose: ", fused_pose)

    # Convert centroids to numpy array for plotting
    # centroids.append(np.array(fused_pose))
    # if fused_pose is None: # TODO: change the logic, to have empty or something printed out not just return. no visible tags
    #     return
    # fused_centroid_path = np.array(fused_centroid_path)

    # print("Fused Centroid Path Length: ", len(fused_centroid_path))
    # print("#######################################")
    # print("Fused Centroid Path x: ", np.array(fused_centroid_path)[:, 0])

    if len(fused_centroid_path) > 0:
        ax_centroid.plot(np.array(fused_centroid_path)[:, 0], np.array(fused_centroid_path)[:, 1], np.array(fused_centroid_path)[:, 2], c='r')
    else:
        pass # or use fused_centroid_path = [[0,0,0]] to plot a point at origin.



    # centroids = fused_pose
    # ax_centroid.scatter(centroids[0], centroids[1], centroids[2], c='r', marker='o', s=50, label='Fused Centroid') # s=5





    # Set labels and title
    ax_centroid.set_xlabel('X (m)')
    ax_centroid.set_ylabel('Y (m)')
    ax_centroid.set_zlabel('Z (m)')
    ax_centroid.set_title('Cube Centroid Path')
    ax_centroid.legend()
    plt.show(block=False)
    plt.pause(0.01)
    
def pose_fusion(poses):  #frame_idx might be needed to be removed evrywhere in this function.
    """
    Function to fuse multiple tag poses to get a more stable pose estimate for the centroid: based on the tags visible.
    Input: list of centroid poses from all visible tags [(frame_idx, tx, ty, tz, qx, qy, qz, qw), ...]
    Output: fused pose (tx, ty, tz, qx, qy, qz, qw)
    """

    if not poses:
        return None

    # frame_idx = poses[0][0]
    translations = np.array([[p[0], p[1], p[2]] for p in poses])
    quaternions = np.array([[p[3], p[4], p[5], p[6]] for p in poses])

    # Average translation
    avg_translation = np.mean(translations, axis=0)

    # Average quaternion (using Singular Value Decomposition method): 
    # Its a least squares solution to find the average quaternion.
    # More robust than simple averaging.
    A = np.zeros((4, 4))
    for q in quaternions:
        A += np.outer(q, q)
    A /= len(quaternions)
    _, _, Vt = np.linalg.svd(A)
    avg_quaternion = Vt[0]

    return ( avg_translation[0], avg_translation[1], avg_translation[2],
            avg_quaternion[0], avg_quaternion[1], avg_quaternion[2], avg_quaternion[3])
    

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

        # Initialize pose filter
        # filter_pose = PoseFilterWindowEMA(window_size=5, alpha_pos=0.2, alpha_rot=0.2)

        # Data struct to have all visible tags and poses in this frame.
        current_frame_poses = []

        for r in results:
            tag_id = r.tag_id
            # current_frame_pose.append(tag_id)
            

            # Compute pose using intrinsics and tag size
            M, init_error, final_error = detector.detection_pose(r, [fx, fy, cx, cy], tag_size)
            t = M[:3, 3]  # translation
            R_mat = M[:3, :3]  # rotation
            quat = R.from_matrix(R_mat).as_quat()  # (x, y, z, w)

            pose = (frame_idx, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
            current_frame_pose = (tag_id, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3])
            current_frame_poses.append(current_frame_pose)
            # pose = filter_pose.filter(tag_id, pose)
            tag_paths.setdefault(tag_id, []).append(pose)
            
            # Draw tag outline
            ptA, ptB, ptC, ptD = map(tuple, r.corners)
            ptA, ptB, ptC, ptD = tuple(map(int, ptA)), tuple(map(int, ptB)), tuple(map(int, ptC)), tuple(map(int, ptD))
            cX, cY = map(int, r.center)

            cv2.line(color_image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(color_image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(color_image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(color_image, ptD, ptA, (0, 255, 0), 2)
            cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)  # Individual centre points.

            # # Project the 3D centroid to 2D image coordinates [ Pinhole camera model ] : If ever i decide to visualise the centroid on the image: not intuitive much though.
            # centroid_2d = (
            #     int(cx + fx * cube_centroid[0] / cube_centroid[2]),
            #     int(cy + fy * cube_centroid[1] / cube_centroid[2]),
            # )
            # cv2.circle(color_image, centroid_2d, 10, (255, 0, 0), -1)

            cv2.putText(color_image, f"ID: {tag_id}", (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        # Plot centroid based on this tag's pose
        plot_centroid_path(ax_centroid, current_frame_poses)

        # Save tag paths to csv every 100 frames, plot and save video at the end.
        if frame_idx % 10 == 0 and frame_idx > 0:
            with open('tag_paths.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['tag_id', 'frame_idx', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
                for tag_id, poses in tag_paths.items():
                    for pose in poses:
                        writer.writerow([tag_id] + list(pose))            
            plot_6d_pose(ax)
            # for relative poses in int tag part.
            plot_3d_relative_pose(ax_relative)             


        # Visual: combine color + depth for display
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        if depth_colormap.shape != color_image.shape:
            color_resized = cv2.resize(color_image, (depth_colormap.shape[1], depth_colormap.shape[0]))
            display_image = np.hstack((color_resized, depth_colormap))
        else:
            display_image = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense + AprilTags", display_image)
        frames.append(display_image)

        
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        frame_idx += 1

finally:

     # Save final plotted image.
    plt.savefig('6d_pose_paths.png')
    # plt.savefig('6d_pose_paths.png', dpi=300, bbox_inches='tight')


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
# 1. dynamic plotting of 6d pose path every 100 frames: Done
# 4. Filtering of poses: clustering etc stuff, so at same no movement pose, the tracking is stable.
# 2. saving the static tag and calculate relative pose of dynamic tags to the static tag.
# 3. transforming id based locations.

# Kal give stuff for printing, and then complete this: and record a video showing pura rotate karne pe pose is same, and translate karne pe reflectted same as it physically moves.
# then start with wisker movements.: fan stuff:   do it on fri and saturday!!