#!/usr/bin/env python3
"""
Motion visualization using matplotlib.

HY-Motion keypoints3d coordinate system:
  X: forward/back
  Y: left/right
  Z: up (vertical)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import argparse


# Kinematic chains for 22-joint SMPL skeleton
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],       # Right leg
    [0, 1, 4, 7, 10],       # Left leg
    [0, 3, 6, 9, 12, 15],   # Spine
    [9, 14, 17, 19, 21],    # Right arm
    [9, 13, 16, 18, 20],    # Left arm
]

COLORS = ["#DD5A37", "#D69E00", "#61CEB9", "#34C1E2", "#80B79A"]


def load_motion(npz_path: str) -> tuple[np.ndarray, int]:
    """Load motion data from npz file."""
    data = np.load(npz_path, allow_pickle=True)

    if 'keypoints3d' in data:
        positions = data['keypoints3d']
    elif 'joint_positions' in data:
        positions = data['joint_positions']
        if positions.ndim == 2:
            num_frames = positions.shape[0]
            num_joints = positions.shape[1] // 3
            positions = positions.reshape(num_frames, num_joints, 3)
    else:
        raise ValueError(f"No joint position data found. Keys: {list(data.keys())}")

    fps = int(data.get('fps', 30))
    return positions, fps


def visualize_motion(positions: np.ndarray, fps: int = 30, speed: float = 1.0, radius: float = 1.5):
    """
    Animate the motion in a 3D plot.

    Data coordinate system: X=forward, Y=left/right, Z=up
    Display: X=left/right, Y=forward, Z=up
    """
    matplotlib.use('TkAgg')

    num_frames, num_joints, _ = positions.shape
    data = positions.copy()

    # Data has person lying along X axis (head at -X, feet at +X)
    # Rotate 90° around Y axis to stand up: new_Z = -old_X
    rotated = np.zeros_like(data)
    rotated[:, :, 0] = data[:, :, 2]   # new X = old Z
    rotated[:, :, 1] = data[:, :, 1]   # new Y = old Y (unchanged)
    rotated[:, :, 2] = -data[:, :, 0]  # new Z = -old X (up)
    data = rotated

    # Only use body joints (first 22)
    kinematic_chain = [
        [j for j in chain if j < num_joints]
        for chain in t2m_kinematic_chain
    ]
    kinematic_chain = [c for c in kinematic_chain if len(c) > 1]

    print(f"Using {len(kinematic_chain)} kinematic chains for {num_joints} joints")

    # After rotation, Z is up - offset so feet touch ground (Z=0)
    min_z = data[:, :22, 2].min()  # Only body joints
    data[:, :, 2] -= min_z

    # Set up figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute data range
    data_range = max(
        data[:, :, 0].max() - data[:, :, 0].min(),
        data[:, :, 1].max() - data[:, :, 1].min(),
        data[:, :, 2].max() - data[:, :, 2].min()
    ) * 0.6

    def update(frame):
        ax.clear()

        # Get current frame, center horizontally on pelvis
        frame_data = data[frame].copy()
        pelvis = frame_data[0]
        frame_data[:, 0] -= pelvis[0]  # Center X (side)
        frame_data[:, 1] -= pelvis[1]  # Center Y (forward)

        # Plot skeleton - after rotation: X=side, Y=forward, Z=up
        for i, chain in enumerate(kinematic_chain):
            color = COLORS[i % len(COLORS)]
            chain_data = frame_data[chain]
            ax.plot3D(
                chain_data[:, 0],  # X (side)
                chain_data[:, 1],  # Y (forward)
                chain_data[:, 2],  # Z (up)
                linewidth=4.0,
                color=color
            )

        # Plot joints
        ax.scatter(
            frame_data[:22, 0],
            frame_data[:22, 1],
            frame_data[:22, 2],
            c='black', s=30
        )

        # Set axis limits
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_zlim3d([0, radius * 2])

        ax.set_xlabel('X (side)')
        ax.set_ylabel('Y (forward)')
        ax.set_zlabel('Z (up)')

        # View angle
        ax.view_init(elev=15, azim=-60)

        ax.set_title(f'Frame {frame}/{num_frames} ({frame/fps:.2f}s)')
        return []

    interval = 1000 / (fps * speed)
    anim = FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval, blit=False, repeat=True
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize motion from NPZ file')
    parser.add_argument('input', help='Input NPZ file with motion data')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed')
    parser.add_argument('--radius', type=float, default=1.0, help='View radius')
    args = parser.parse_args()

    print(f"Loading motion from {args.input}...")
    positions, fps = load_motion(args.input)
    print(f"  Frames: {positions.shape[0]}, Joints: {positions.shape[1]}, FPS: {fps}")

    visualize_motion(positions, fps, args.speed, args.radius)


if __name__ == '__main__':
    main()
