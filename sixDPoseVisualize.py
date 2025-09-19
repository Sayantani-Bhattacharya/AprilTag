import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


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
                float(row['tz'])
                # float(row['qx']),
                # float(row['qy']),
                # float(row['qz']),
                # float(row['qw'])
            )
            tag_paths.setdefault(tag_id, []).append(pose)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Plot each tag's path
    for tag_id, poses in tag_paths.items():
        poses = sorted(poses, key=lambda x: x[0])  # sort by frame_idx
        tx = [p[1] for p in poses]
        ty = [p[2] for p in poses]
        tz = [p[3] for p in poses]
        ax.plot(tx, ty, tz, label=f'Tag {tag_id}')

        # # Plot orientation as arrows every N frames.
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
    plt.title('AprilTag 6D Pose Paths')
    plt.show(block=False)
    plt.pause(0.01)
    # plt.savefig('6d_pose_paths.png', dpi=300, bbox_inches='tight')





