"""
SMPL-H to Mixamo Skeleton Retargeting

Converts SMPL-H motion data to Mixamo-compatible skeleton format.
This enables using HY-Motion output with any Mixamo-rigged character.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Optional


# SMPL-H joint indices (22 body joints + hands)
SMPLH_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

# Mixamo skeleton joint names
MIXAMO_JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "HeadTop_End",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]

# SMPL-H to Mixamo joint mapping
SMPLH_TO_MIXAMO = {
    "pelvis": "Hips",
    "spine1": "Spine",
    "spine2": "Spine1",
    "spine3": "Spine2",
    "neck": "Neck",
    "head": "Head",
    "left_collar": "LeftShoulder",
    "left_shoulder": "LeftArm",
    "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand",
    "right_collar": "RightShoulder",
    "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm",
    "right_wrist": "RightHand",
    "left_hip": "LeftUpLeg",
    "left_knee": "LeftLeg",
    "left_ankle": "LeftFoot",
    "left_foot": "LeftToeBase",
    "right_hip": "RightUpLeg",
    "right_knee": "RightLeg",
    "right_ankle": "RightFoot",
    "right_foot": "RightToeBase",
}

# Rotation corrections for coordinate system differences
# SMPL-H uses Y-up, Z-forward; Mixamo typically uses Y-up, Z-back
AXIS_CORRECTIONS = {
    "Hips": np.array([0, 0, 0]),  # Root - no correction needed
    "Spine": np.array([0, 0, 0]),
    "LeftArm": np.array([0, 0, -90]),  # Arm T-pose adjustment
    "RightArm": np.array([0, 0, 90]),
    "LeftUpLeg": np.array([0, 0, 0]),
    "RightUpLeg": np.array([0, 0, 0]),
}


@dataclass
class MotionData:
    """Container for motion data."""

    rotations: np.ndarray  # (num_frames, num_joints, 3) euler or (num_frames, num_joints, 4) quat
    root_positions: np.ndarray  # (num_frames, 3)
    fps: int = 30
    rotation_format: str = "euler"  # "euler" or "quaternion"


@dataclass
class RetargetedMotion:
    """Container for Mixamo-retargeted motion."""

    joint_rotations: dict[str, np.ndarray]  # joint_name -> (num_frames, 3) euler angles
    root_positions: np.ndarray  # (num_frames, 3)
    fps: int = 30


def axis_angle_to_euler(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation to euler angles (XYZ order)."""
    if axis_angle.shape[-1] == 3:
        rotation = Rotation.from_rotvec(axis_angle)
    else:
        rotation = Rotation.from_quat(axis_angle)
    return rotation.as_euler("xyz", degrees=True)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles (degrees) to quaternion."""
    rotation = Rotation.from_euler("xyz", euler, degrees=True)
    return rotation.as_quat()


def apply_rotation_correction(
    rotation: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    """Apply axis correction to rotation."""
    if np.all(correction == 0):
        return rotation

    rot = Rotation.from_euler("xyz", rotation, degrees=True)
    corr = Rotation.from_euler("xyz", correction, degrees=True)
    result = corr * rot
    return result.as_euler("xyz", degrees=True)


def retarget_smplh_to_mixamo(
    body_pose: np.ndarray,
    global_orient: Optional[np.ndarray] = None,
    transl: Optional[np.ndarray] = None,
    fps: int = 30,
) -> RetargetedMotion:
    """
    Retarget SMPL-H motion data to Mixamo skeleton.

    Args:
        body_pose: SMPL-H body pose parameters (num_frames, 63) or (num_frames, 21, 3)
                   Axis-angle rotations for 21 body joints (excluding pelvis)
        global_orient: Global orientation (num_frames, 3), axis-angle for pelvis
        transl: Root translation (num_frames, 3)
        fps: Frames per second

    Returns:
        RetargetedMotion with Mixamo joint rotations and root positions
    """
    num_frames = body_pose.shape[0]

    # Reshape body_pose if needed
    if body_pose.ndim == 2 and body_pose.shape[1] == 63:
        body_pose = body_pose.reshape(num_frames, 21, 3)

    # Initialize output
    joint_rotations = {joint: np.zeros((num_frames, 3)) for joint in MIXAMO_JOINTS}

    # Handle global orientation (pelvis/Hips)
    if global_orient is not None:
        if global_orient.ndim == 2 and global_orient.shape[1] == 3:
            hips_euler = np.array([axis_angle_to_euler(go) for go in global_orient])
        else:
            hips_euler = global_orient
        joint_rotations["Hips"] = hips_euler
    else:
        joint_rotations["Hips"] = np.zeros((num_frames, 3))

    # Map SMPL-H joints to Mixamo
    for smpl_name, smpl_idx in SMPLH_JOINTS.items():
        if smpl_name == "pelvis":
            continue  # Already handled via global_orient

        if smpl_name in SMPLH_TO_MIXAMO:
            mixamo_name = SMPLH_TO_MIXAMO[smpl_name]

            # Get joint index in body_pose (pelvis is not in body_pose)
            joint_idx = smpl_idx - 1  # Offset by 1 since pelvis is not in body_pose

            if joint_idx < 0 or joint_idx >= body_pose.shape[1]:
                continue

            # Convert axis-angle to euler
            joint_aa = body_pose[:, joint_idx, :]
            joint_euler = np.array([axis_angle_to_euler(aa) for aa in joint_aa])

            # Apply axis corrections if needed
            correction = AXIS_CORRECTIONS.get(mixamo_name, np.array([0, 0, 0]))
            if not np.all(correction == 0):
                joint_euler = np.array([
                    apply_rotation_correction(e, correction) for e in joint_euler
                ])

            joint_rotations[mixamo_name] = joint_euler

    # Handle root translation
    if transl is not None:
        root_positions = transl.copy()
        # Scale if needed (SMPL uses meters, game engines often use different scales)
        # Adjust this multiplier based on your target engine
        root_positions *= 100  # Convert to centimeters (common for Unreal/Unity)
    else:
        root_positions = np.zeros((num_frames, 3))

    return RetargetedMotion(
        joint_rotations=joint_rotations,
        root_positions=root_positions,
        fps=fps,
    )


def retarget_from_motion_dict(
    motion_data: dict,
    fps: int = 30,
) -> RetargetedMotion:
    """
    Retarget motion from dictionary format (as returned by RunPod handler).

    Args:
        motion_data: Dictionary with 'body_pose', 'global_orient', 'transl' keys
        fps: Frames per second

    Returns:
        RetargetedMotion with Mixamo joint rotations
    """
    body_pose = motion_data.get("body_pose")
    global_orient = motion_data.get("global_orient")
    transl = motion_data.get("transl")

    # Handle case where motion is stored as single 'motion' key
    if body_pose is None and "motion" in motion_data:
        motion = motion_data["motion"]
        # Try to split into body_pose and global_orient
        if motion.ndim == 3 and motion.shape[-1] == 3:
            # Assume first joint is global orient, rest is body pose
            global_orient = motion[:, 0, :]
            body_pose = motion[:, 1:22, :]  # Take 21 body joints

    if body_pose is None:
        raise ValueError("No body_pose or motion data found in input")

    return retarget_smplh_to_mixamo(
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        fps=fps,
    )
