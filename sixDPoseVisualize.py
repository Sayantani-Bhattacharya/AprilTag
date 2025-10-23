import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


# External Cube Dimensions
cube_size = 0.166  # meters   

static_tag_id = 15  # Mid of the wisker arrays in seal head.

# Defining rigid transformation matrix from each id to centroid [ Assuming face 1 as front]
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

# Relative poses global variable
tx_rel = []
ty_rel = []
tz_rel = []


def plot_6d_pose(ax):
    ax.clear()  # Clear the previous plot
    # Read CSV and group by tag_id
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            pose = (
                int(row['frame_idx']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(tag_id, []).append(pose)

    # Plot each tag's path
    for tag_id, poses in tag_paths.items():
        poses = sorted(poses, key=lambda x: x[0])  # sort by frame_idx
        tx = [p[1] for p in poses]
        ty = [p[2] for p in poses]
        tz = [p[3] for p in poses]
        ax.plot(tx, ty, tz, label=f'Tag {tag_id}')

        # Plot orientation as arrows every N frames.
        # N = max(1, len(poses)//20)
        # for i in range(0, len(poses), N):
        #     t = np.array([tx[i], ty[i], tz[i]])
        #     quat = poses[i][4:8]
        #     rot = R.from_quat(quat)
        #     # Arrow in the direction of the tag's z-axis
        #     dir_vec = rot.apply([0, 0, 0.05])  # scale for visibility
        #     ax.quiver(t[0], t[1], t[2], dir_vec[0], dir_vec[1], dir_vec[2], color='k', length=0.05, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('AprilTag 6D Pose Paths')
    plt.show(block=False)
    plt.pause(0.01)

def plot_3d_relative_pose(ax_relative):
    ax_relative.clear()  # Clear the previous plot
    # Read CSV and group by tag_id
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag_id = int(row['tag_id'])
            pose = (
                int(row['frame_idx']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(tag_id, []).append(pose)

    # Extract pose with ID = 0 as reference
    if static_tag_id not in tag_paths:
        print("No tag with ID static_tag_id found for reference.")
        return
    ref_poses = sorted(tag_paths[0], key=lambda x: x[0])  # sort by frame_idx

    # Plot each tag's path relative to tag 0: as in x-x1, y-y1, z-z1

    for tag_id, poses in tag_paths.items():
        if tag_id == static_tag_id:
            continue
        poses = sorted(poses, key=lambda x: x[0])
        tx_rel = []
        ty_rel = []
        tz_rel = []

        for p in poses:
            frame_idx = p[0]
            ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
            if ref_pose is None:
                continue

            # Extract translation poses
            t = np.array([p[1], p[2], p[3]])
            t_ref = np.array([ref_pose[1], ref_pose[2], ref_pose[3]])
            # Extract rotation poses
            R = R.from_quat(p[4:8]).as_matrix()
            R_ref = R.from_quat(ref_pose[4:8]).as_matrix()

            # Compute relative transformation
            R_rel =  R_ref.T @ R
            t_rel = R_ref.T @ (t - t_ref)

            tx_rel.append(t_rel[0])
            ty_rel.append(t_rel[1])
            tz_rel.append(t_rel[2])      
        
        ax_relative.plot(tx_rel, ty_rel, tz_rel, label=f'Tag {tag_id}')
    
    ax_relative.set_xlabel('X rel to Static Tag')
    ax_relative.set_ylabel('Y rel to Static Tag')
    ax_relative.set_zlabel('Z rel to Static Tag')
    ax_relative.legend()
    ax_relative.set_title('Relative 3D Poses to Static Tag')
    plt.show(block=False)
    plt.pause(0.01)

def plot_relative_pose_indv(tag_id, ax_indv):
    # Plot the relative pose of a specific tag to Static Tag over time as a path
    ax_indv.clear()  # Clear the previous plot
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_idx']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if static_tag_id not in tag_paths or tag_id not in tag_paths:
        print(f"Static Tag or Tag {tag_id} not found.")
        return
    ref_poses = sorted(tag_paths[0], key=lambda x: x[0])
    poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
    # tx_rel = []
    # ty_rel = []
    # tz_rel = []

    for p in poses:
        frame_idx = p[0]
        ref_pose = next((rp for rp in ref_poses if rp[0] == frame_idx), None)
        if ref_pose is None:
            continue

        # Extract translation poses
        t = np.array([p[1], p[2], p[3]])
        t_ref = np.array([ref_pose[1], ref_pose[2], ref_pose[3]])
        # Extract rotation poses
        R = R.from_quat(p[4:8]).as_matrix()
        R_ref = R.from_quat(ref_pose[4:8]).as_matrix()

        # Compute relative transformation
        R_rel =  R_ref.T @ R
        t_rel = R_ref.T @ (t - t_ref)

        tx_rel.append(t_rel[0])
        ty_rel.append(t_rel[1])
        tz_rel.append(t_rel[2])
    
    
    ax_indv.plot(tx_rel, ty_rel, tz_rel, label=f'Relative Path of Tag {tag_id}')
    ax_indv.set_xlabel('X rel to Static Tag')
    ax_indv.set_ylabel('Y rel to Static Tag')
    ax_indv.set_zlabel('Z rel to Static Tag')
    ax_indv.legend()
    ax_indv.set_title(f'Relative Path of Tag {tag_id} to Static Tag')
    plt.show(block=False)
    plt.pause(0.01)

def plot_idv(tag_id, ax_indv_axis, ax_x = None, ax_y= None, ax_z= None):
    # Plot the individual x, y, z signals of a specific tag separately over time.
    ax_indv_axis.clear()  # Clear the previous plot
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_idx']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if tag_id not in tag_paths:
        print(f"Tag {tag_id} not found.")
        return
    poses = sorted(tag_paths[tag_id], key=lambda x: x[0])
    frame_idxs = [p[0] for p in poses]
    tx = [p[1] for p in poses]
    ty = [p[2] for p in poses]
    tz = [p[3] for p in poses]

    # # Create subplots for x, y, z signals
    # fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(8, 12))
    # fig.suptitle(f'Separate Axes of Tag {tag_id} Over Time')

    # if (not None all then do this)


    # Plot X signal
    ax_x.plot(frame_idxs, tx, label='X', color='r')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y signal
    ax_y.plot(frame_idxs, ty, label='Y', color='g')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z signal
    ax_z.plot(frame_idxs, tz, label='Z', color='b')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # plt.pause(0.01)

def plot_idv_denoised(moving_tag_id, static_tag_id, ax_indv_axis, ax_x = None, ax_y= None, ax_z= None):
    # Plot the individual x, y, z signals of a specific tag separately over time.
    ax_indv_axis.clear()  # Clear the previous plot
    tag_paths = {}
    with open('tag_paths.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_id = int(row['tag_id'])
            pose = (
                int(row['frame_idx']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz']),
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            )
            tag_paths.setdefault(t_id, []).append(pose)
    if moving_tag_id not in tag_paths or static_tag_id not in tag_paths:
        print(f"Tag {moving_tag_id} or Static Tag {static_tag_id} not found.")
        return
    moving_poses = sorted(tag_paths[moving_tag_id], key=lambda x: x[0])
    static_poses = sorted(tag_paths[static_tag_id], key=lambda x: x[0])

    frame_idxs = []
    tx_rel = []
    ty_rel = []
    tz_rel = []

    for p in moving_poses:
        frame_idx = p[0]
        static_pose = next((sp for sp in static_poses if sp[0] == frame_idx), None)
        if static_pose is None:
            continue
        t_moving = np.array([p[1], p[2], p[3]])
        t_static = np.array([static_pose[1], static_pose[2], static_pose[3]])
        frame_idxs.append(frame_idx)
        tx_rel.append(t_moving[0] - t_static[0])
        ty_rel.append(t_moving[1] - t_static[1])
        tz_rel.append(t_moving[2] - t_static[2])

    # Plot X signal
    ax_x.plot(frame_idxs, tx_rel, label='X rel to Static Tag', color='r')
    ax_x.set_xlabel('Frame Index')
    ax_x.set_ylabel('X Position (m)')
    ax_x.legend()
    ax_x.grid()

    # Plot Y signal
    ax_y.plot(frame_idxs, ty_rel, label='Y rel to Static Tag', color='g')
    ax_y.set_xlabel('Frame Index')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.legend()
    ax_y.grid()

    # Plot Z signal
    ax_z.plot(frame_idxs, tz_rel, label='Z rel to Static Tag', color='b')
    ax_z.set_xlabel('Frame Index')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.legend()
    ax_z.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # plt.pause(0.01)

def analysis_metrics():
    pass
    # orientation, vel stuff, pca, fft..
    



if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 6))

    # ax1 = fig.add_subplot(131, projection='3d')
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax3 = fig.add_subplot(133, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')

    # Create subplots for x, y, z signals
    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(8, 12))
    fig.suptitle(f'Separate Axes of Tag Over Time')

    while True:
        # plot_6d_pose(ax1)
        # plot_3d_relative_pose(ax2)
        # plot_relative_pose_indv(1, ax3)
        # plot_idv(1, ax4)

        plot_idv(19, ax5, ax_x, ax_y, ax_z)                 ## x,y,z signals wrt time for a tag.

        # plot_idv_denoised( 19, 1, ax5, ax_x, ax_y, ax_z)  ## x,y,z signals wrt time for a tag denoised by static tag








# TODO: the denoising is not in reference to the static tag need to do that!!


# To Note IMP:  
# before usinf the ratationn part for the static tag relaative transform, and after the plot remains the same : Wierd, considering 45 degree tilt not same plane...??


























#  Tests
# 1. Oscilate a wisker and have it stationary : compare the 2 signals over time and see if they match
# 2. do flow like motions on one side and see if i can get some info from them