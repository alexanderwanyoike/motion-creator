"""
FBX Export for Mixamo-retargeted Motion

Exports retargeted motion data to FBX format compatible with
Mixamo-rigged characters (e.g., flip-frenzy).

Requires Autodesk FBX SDK Python bindings.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

from retarget import RetargetedMotion, MIXAMO_JOINTS

# Try to import FBX SDK
try:
    import fbx
    from fbx import FbxManager, FbxScene, FbxNode, FbxSkeleton, FbxAnimStack
    from fbx import FbxAnimLayer, FbxAnimCurve, FbxAnimCurveNode, FbxTime
    from fbx import FbxIOSettings, FbxExporter

    HAS_FBX = True
except ImportError:
    HAS_FBX = False
    print("Warning: FBX SDK not found. Install it from Autodesk.")
    print("See: https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3")


# Mixamo skeleton hierarchy
MIXAMO_HIERARCHY = {
    "Hips": None,  # Root
    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",
    "HeadTop_End": "Head",
    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandThumb1": "LeftHand",
    "LeftHandThumb2": "LeftHandThumb1",
    "LeftHandThumb3": "LeftHandThumb2",
    "LeftHandIndex1": "LeftHand",
    "LeftHandIndex2": "LeftHandIndex1",
    "LeftHandIndex3": "LeftHandIndex2",
    "LeftHandMiddle1": "LeftHand",
    "LeftHandMiddle2": "LeftHandMiddle1",
    "LeftHandMiddle3": "LeftHandMiddle2",
    "LeftHandRing1": "LeftHand",
    "LeftHandRing2": "LeftHandRing1",
    "LeftHandRing3": "LeftHandRing2",
    "LeftHandPinky1": "LeftHand",
    "LeftHandPinky2": "LeftHandPinky1",
    "LeftHandPinky3": "LeftHandPinky2",
    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandThumb1": "RightHand",
    "RightHandThumb2": "RightHandThumb1",
    "RightHandThumb3": "RightHandThumb2",
    "RightHandIndex1": "RightHand",
    "RightHandIndex2": "RightHandIndex1",
    "RightHandIndex3": "RightHandIndex2",
    "RightHandMiddle1": "RightHand",
    "RightHandMiddle2": "RightHandMiddle1",
    "RightHandMiddle3": "RightHandMiddle2",
    "RightHandRing1": "RightHand",
    "RightHandRing2": "RightHandRing1",
    "RightHandRing3": "RightHandRing2",
    "RightHandPinky1": "RightHand",
    "RightHandPinky2": "RightHandPinky1",
    "RightHandPinky3": "RightHandPinky2",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}


def check_fbx_available() -> bool:
    """Check if FBX SDK is available."""
    return HAS_FBX


class FBXExporter:
    """Exports motion data to FBX format."""

    def __init__(self):
        if not HAS_FBX:
            raise RuntimeError(
                "FBX SDK not available. Please install it from Autodesk:\n"
                "https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3"
            )

        self.manager = FbxManager.Create()
        if not self.manager:
            raise RuntimeError("Failed to create FBX Manager")

        # Create IO settings
        ios = FbxIOSettings.Create(self.manager, "IOSRoot")
        self.manager.SetIOSettings(ios)

    def __del__(self):
        if hasattr(self, "manager") and self.manager:
            self.manager.Destroy()

    def create_skeleton(self, scene: "FbxScene") -> dict[str, "FbxNode"]:
        """Create Mixamo skeleton hierarchy in the scene."""
        nodes = {}

        # Create nodes in hierarchy order
        for joint_name in MIXAMO_JOINTS:
            # Create skeleton attribute
            skeleton_attr = FbxSkeleton.Create(scene, joint_name)

            if joint_name == "Hips":
                skeleton_attr.SetSkeletonType(FbxSkeleton.EType.eRoot)
            elif joint_name.endswith("_End"):
                skeleton_attr.SetSkeletonType(FbxSkeleton.EType.eEffector)
            else:
                skeleton_attr.SetSkeletonType(FbxSkeleton.EType.eLimbNode)

            # Create node
            node = FbxNode.Create(scene, joint_name)
            node.SetNodeAttribute(skeleton_attr)

            # Set default T-pose position (simplified - would need actual bone lengths)
            # These are approximate positions for a standard humanoid
            self._set_tpose_position(node, joint_name)

            nodes[joint_name] = node

        # Build hierarchy
        for joint_name, parent_name in MIXAMO_HIERARCHY.items():
            if parent_name is None:
                # Root node - add to scene
                scene.GetRootNode().AddChild(nodes[joint_name])
            else:
                if parent_name in nodes:
                    nodes[parent_name].AddChild(nodes[joint_name])

        return nodes

    def _set_tpose_position(self, node: "FbxNode", joint_name: str):
        """Set T-pose position for a joint."""
        # Simplified T-pose positions (in centimeters)
        positions = {
            "Hips": (0, 100, 0),
            "Spine": (0, 110, 0),
            "Spine1": (0, 120, 0),
            "Spine2": (0, 130, 0),
            "Neck": (0, 145, 0),
            "Head": (0, 155, 0),
            "HeadTop_End": (0, 170, 0),
            "LeftShoulder": (5, 140, 0),
            "LeftArm": (20, 140, 0),
            "LeftForeArm": (45, 140, 0),
            "LeftHand": (70, 140, 0),
            "RightShoulder": (-5, 140, 0),
            "RightArm": (-20, 140, 0),
            "RightForeArm": (-45, 140, 0),
            "RightHand": (-70, 140, 0),
            "LeftUpLeg": (10, 95, 0),
            "LeftLeg": (10, 50, 0),
            "LeftFoot": (10, 5, 5),
            "LeftToeBase": (10, 0, 15),
            "RightUpLeg": (-10, 95, 0),
            "RightLeg": (-10, 50, 0),
            "RightFoot": (-10, 5, 5),
            "RightToeBase": (-10, 0, 15),
        }

        if joint_name in positions:
            x, y, z = positions[joint_name]
            node.LclTranslation.Set(fbx.FbxDouble3(x, y, z))

    def add_animation(
        self,
        scene: "FbxScene",
        nodes: dict[str, "FbxNode"],
        motion: RetargetedMotion,
        anim_name: str = "motion",
    ):
        """Add animation data to the skeleton."""
        # Create animation stack
        anim_stack = FbxAnimStack.Create(scene, anim_name)
        anim_layer = FbxAnimLayer.Create(scene, "Base Layer")
        anim_stack.AddMember(anim_layer)

        num_frames = motion.root_positions.shape[0]
        fps = motion.fps

        # Create time for keyframes
        time = FbxTime()

        # Animate each joint
        for joint_name, rotations in motion.joint_rotations.items():
            if joint_name not in nodes:
                continue

            node = nodes[joint_name]

            # Create animation curves for rotation
            curve_node = node.LclRotation.GetCurveNode(anim_layer, True)

            curve_x = node.LclRotation.GetCurve(anim_layer, "X", True)
            curve_y = node.LclRotation.GetCurve(anim_layer, "Y", True)
            curve_z = node.LclRotation.GetCurve(anim_layer, "Z", True)

            curve_x.KeyModifyBegin()
            curve_y.KeyModifyBegin()
            curve_z.KeyModifyBegin()

            for frame_idx in range(num_frames):
                time.SetSecondDouble(frame_idx / fps)

                rx, ry, rz = rotations[frame_idx]

                key_index = curve_x.KeyAdd(time)[0]
                curve_x.KeySetValue(key_index, rx)
                curve_x.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationCubic)

                key_index = curve_y.KeyAdd(time)[0]
                curve_y.KeySetValue(key_index, ry)
                curve_y.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationCubic)

                key_index = curve_z.KeyAdd(time)[0]
                curve_z.KeySetValue(key_index, rz)
                curve_z.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationCubic)

            curve_x.KeyModifyEnd()
            curve_y.KeyModifyEnd()
            curve_z.KeyModifyEnd()

        # Animate root position (Hips)
        if "Hips" in nodes:
            hips = nodes["Hips"]

            curve_tx = hips.LclTranslation.GetCurve(anim_layer, "X", True)
            curve_ty = hips.LclTranslation.GetCurve(anim_layer, "Y", True)
            curve_tz = hips.LclTranslation.GetCurve(anim_layer, "Z", True)

            curve_tx.KeyModifyBegin()
            curve_ty.KeyModifyBegin()
            curve_tz.KeyModifyBegin()

            for frame_idx in range(num_frames):
                time.SetSecondDouble(frame_idx / fps)

                tx, ty, tz = motion.root_positions[frame_idx]

                key_index = curve_tx.KeyAdd(time)[0]
                curve_tx.KeySetValue(key_index, tx)

                key_index = curve_ty.KeyAdd(time)[0]
                curve_ty.KeySetValue(key_index, ty)

                key_index = curve_tz.KeyAdd(time)[0]
                curve_tz.KeySetValue(key_index, tz)

            curve_tx.KeyModifyEnd()
            curve_ty.KeyModifyEnd()
            curve_tz.KeyModifyEnd()

    def export(
        self,
        motion: RetargetedMotion,
        output_path: str | Path,
        anim_name: str = "motion",
    ) -> bool:
        """
        Export motion to FBX file.

        Args:
            motion: RetargetedMotion with Mixamo joint rotations
            output_path: Output FBX file path
            anim_name: Name for the animation take

        Returns:
            True if export successful, False otherwise
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create scene
        scene = FbxScene.Create(self.manager, "MotionScene")

        # Set scene info
        scene_info = scene.GetSceneInfo()
        if scene_info:
            scene_info.mTitle = anim_name
            scene_info.mAuthor = "HY-Motion Pipeline"

        # Create skeleton
        nodes = self.create_skeleton(scene)

        # Add animation
        self.add_animation(scene, nodes, motion, anim_name)

        # Export
        exporter = FbxExporter.Create(self.manager, "")

        # Use binary FBX format
        file_format = self.manager.GetIOPluginRegistry().GetNativeWriterFormat()

        if not exporter.Initialize(str(output_path), file_format, self.manager.GetIOSettings()):
            print(f"Failed to initialize exporter: {exporter.GetStatus().GetErrorString()}")
            return False

        success = exporter.Export(scene)
        exporter.Destroy()
        scene.Destroy()

        return success


def export_motion_to_fbx(
    motion: RetargetedMotion,
    output_path: str | Path,
    anim_name: Optional[str] = None,
) -> bool:
    """
    Export retargeted motion to FBX file.

    Args:
        motion: RetargetedMotion with Mixamo joint rotations
        output_path: Output FBX file path
        anim_name: Optional name for the animation

    Returns:
        True if export successful
    """
    if not HAS_FBX:
        raise RuntimeError("FBX SDK not available")

    output_path = Path(output_path)
    if anim_name is None:
        anim_name = output_path.stem

    exporter = FBXExporter()
    return exporter.export(motion, output_path, anim_name)


def export_motion_to_glb(
    motion: RetargetedMotion,
    output_path: str | Path,
) -> bool:
    """
    Export motion to GLB format (alternative to FBX, no SDK required).

    Uses trimesh for GLB export.

    Args:
        motion: RetargetedMotion with Mixamo joint rotations
        output_path: Output GLB file path

    Returns:
        True if export successful
    """
    try:
        import trimesh
    except ImportError:
        raise RuntimeError("trimesh not available. Install with: pip install trimesh")

    # GLB export implementation would go here
    # This is a simplified placeholder - full implementation would create
    # proper glTF skeleton and animation data

    print("GLB export not yet fully implemented. Use FBX export instead.")
    return False
