"""
Structured Light Projection from Depth Maps

This module provides a depth-map-centric approach to structured light simulation.
It can work with either:
1. Synthetic depth maps generated from 3D geometry
2. External depth maps from real depth cameras (RealSense, Kinect, etc.)

The key difference from structured_light_3d.py is that pattern projection
is decoupled from 3D geometry - it only needs depth maps and camera calibration.
"""

import numpy as np
import pyrender
import trimesh
from typing import Tuple, Optional, Dict, Any
import yaml
from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass
class CameraCalibration:
    """Camera calibration parameters."""
    position: np.ndarray  # 3D position in world coordinates
    look_at: np.ndarray   # Point camera looks at
    fov: float            # Field of view in degrees
    resolution: Tuple[int, int]  # (width, height)

    # Intrinsics (computed from FOV if not provided)
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

    def __post_init__(self):
        """Compute intrinsics from FOV if not provided."""
        if self.fx is None:
            fov_rad = np.radians(self.fov)
            self.fx = self.resolution[0] / (2 * np.tan(fov_rad / 2))
            self.fy = self.resolution[1] / (2 * np.tan(fov_rad / 2))
            self.cx = self.resolution[0] / 2
            self.cy = self.resolution[1] / 2


@dataclass
class ProjectorCalibration:
    """Projector calibration parameters."""
    position: np.ndarray  # 3D position in world coordinates
    look_at: np.ndarray   # Point projector looks at
    fov: float            # Field of view in degrees
    resolution: Tuple[int, int]  # (width, height)

    # Intrinsics (computed from FOV if not provided)
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None

    def __post_init__(self):
        """Compute intrinsics from FOV if not provided."""
        if self.fx is None:
            fov_rad = np.radians(self.fov)
            self.fx = self.resolution[0] / (2 * np.tan(fov_rad / 2))
            self.fy = self.resolution[1] / (2 * np.tan(fov_rad / 2))
            self.cx = self.resolution[0] / 2
            self.cy = self.resolution[1] / 2


class DepthMapGenerator:
    """Generate depth maps from 3D geometry."""

    def __init__(self, camera_calibration: CameraCalibration):
        """
        Initialize depth map generator.

        Args:
            camera_calibration: Camera calibration parameters
        """
        self.camera_calib = camera_calibration

    def render_depth(self, scene) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render depth map from a 3D scene.

        Args:
            scene: Scene3D object (from structured_light_3d.py)

        Returns:
            Tuple of (rgb_image, depth_map)
        """
        # Import here to avoid circular dependency
        from structured_light_3d import Scene3D

        # Create pyrender scene
        pr_scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])

        # Add meshes
        for mesh, pose in zip(scene.meshes, scene.mesh_poses):
            pr_mesh = pyrender.Mesh.from_trimesh(mesh)
            pr_scene.add(pr_mesh, pose=pose)

        # Add camera
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.camera_calib.fov),
            aspectRatio=self.camera_calib.resolution[0] / self.camera_calib.resolution[1]
        )
        camera_pose = self._look_at_matrix(
            self.camera_calib.position,
            self.camera_calib.look_at
        )
        pr_scene.add(camera, pose=camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        pr_scene.add(light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.camera_calib.resolution[0],
            viewport_height=self.camera_calib.resolution[1]
        )
        rgb, depth = renderer.render(pr_scene)
        renderer.delete()

        return rgb, depth

    def _look_at_matrix(self, eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Create look-at transformation matrix (OpenGL convention)."""
        # Camera looks down -Z axis in OpenGL
        forward = eye - target
        forward = forward / np.linalg.norm(forward)

        # Use world up vector
        world_up = np.array([0.0, 0.0, 1.0])

        # Compute right vector
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        # Recompute up
        up = np.cross(forward, right)

        # Create rotation matrix (column vectors)
        rotation = np.column_stack([right, up, forward])

        # Create 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = eye

        return matrix


class PatternProjector:
    """Project structured light patterns onto depth maps."""

    def __init__(self,
                 camera_calibration: CameraCalibration,
                 projector_calibration: ProjectorCalibration):
        """
        Initialize pattern projector.

        Args:
            camera_calibration: Camera calibration parameters
            projector_calibration: Projector calibration parameters
        """
        self.camera_calib = camera_calibration
        self.projector_calib = projector_calibration

    def project_pattern_backward(self,
                                 pattern: np.ndarray,
                                 depth_map: np.ndarray,
                                 base_image: Optional[np.ndarray] = None,
                                 pattern_intensity: float = 0.8) -> np.ndarray:
        """
        Project pattern using backward (pull) projection.
        Good for continuous patterns like stripes.

        Args:
            pattern: Projector pattern (H x W) with values in [0, 1]
            depth_map: Depth map from camera viewpoint (H x W)
            base_image: Optional base RGB image to overlay pattern on
            pattern_intensity: Intensity of pattern overlay

        Returns:
            RGB image with projected pattern
        """
        h, w = self.camera_calib.resolution[1], self.camera_calib.resolution[0]
        pattern_rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Camera intrinsics
        camera_fx = self.camera_calib.fx
        camera_fy = self.camera_calib.fy
        camera_cx = self.camera_calib.cx
        camera_cy = self.camera_calib.cy

        # Camera pose
        camera_pose = self._look_at_matrix(
            self.camera_calib.position,
            self.camera_calib.look_at
        )

        # Projector intrinsics
        proj_fx = self.projector_calib.fx
        proj_fy = self.projector_calib.fy
        proj_cx = self.projector_calib.cx
        proj_cy = self.projector_calib.cy
        proj_w, proj_h = self.projector_calib.resolution

        # Projector pose
        proj_pose = self._look_at_matrix(
            self.projector_calib.position,
            self.projector_calib.look_at
        )
        proj_pose_inv = np.linalg.inv(proj_pose)

        # For each camera pixel
        for v in range(h):
            for u in range(w):
                d = depth_map[v, u]
                if d <= 0:
                    continue

                # Back-project to 3D world (OpenGL: camera looks down -Z)
                x_cam = (u - camera_cx) * d / camera_fx
                y_cam = (v - camera_cy) * d / camera_fy
                z_cam = -d  # Negative because camera looks down -Z

                # Transform to world coordinates
                point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                point_world = camera_pose @ point_cam

                # Transform to projector coordinates
                point_proj = proj_pose_inv @ point_world

                # Project onto projector image plane
                if point_proj[2] < 0:  # In front of projector (negative Z)
                    proj_u = proj_fx * point_proj[0] / point_proj[2] + proj_cx
                    proj_v = proj_fy * point_proj[1] / point_proj[2] + proj_cy

                    # Bilinear interpolation
                    if 0 <= proj_u < proj_w - 1 and 0 <= proj_v < proj_h - 1:
                        u0 = int(np.floor(proj_u))
                        v0 = int(np.floor(proj_v))
                        u1 = u0 + 1
                        v1 = v0 + 1

                        wu = proj_u - u0
                        wv = proj_v - v0

                        pattern_value = (
                            pattern[v0, u0] * (1 - wu) * (1 - wv) +
                            pattern[v0, u1] * wu * (1 - wv) +
                            pattern[v1, u0] * (1 - wu) * wv +
                            pattern[v1, u1] * wu * wv
                        )
                        pattern_rgb[v, u, :] = pattern_value

        # Combine with base image if provided
        if base_image is not None:
            base_float = base_image.astype(np.float32) / 255.0
            result = base_float + pattern_rgb * pattern_intensity
            result = np.clip(result, 0, 1)
            return (result * 255).astype(np.uint8)
        else:
            # Create grayscale base from depth
            depth_normalized = np.clip(depth_map / np.max(depth_map), 0, 1)
            base = np.stack([depth_normalized] * 3, axis=-1)
            result = base + pattern_rgb * pattern_intensity
            result = np.clip(result, 0, 1)
            return (result * 255).astype(np.uint8)

    def _look_at_matrix(self, eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Create look-at transformation matrix (OpenGL convention)."""
        forward = eye - target
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation = np.column_stack([right, up, forward])

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = eye

        return matrix


class DepthMapIO:
    """Utilities for loading and saving depth maps."""

    @staticmethod
    def save_depth_png(depth: np.ndarray, filepath: str, max_depth: float = 10.0):
        """
        Save depth map as 16-bit PNG.

        Args:
            depth: Depth map array
            filepath: Output file path
            max_depth: Maximum depth value for normalization
        """
        # Normalize to 16-bit range
        depth_normalized = np.clip(depth / max_depth, 0, 1)
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        cv2.imwrite(filepath, depth_16bit)

    @staticmethod
    def load_depth_png(filepath: str, max_depth: float = 10.0) -> np.ndarray:
        """
        Load depth map from 16-bit PNG.

        Args:
            filepath: Input file path
            max_depth: Maximum depth value for denormalization

        Returns:
            Depth map in meters
        """
        depth_16bit = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        depth = (depth_16bit.astype(np.float32) / 65535.0) * max_depth
        return depth

    @staticmethod
    def save_depth_npy(depth: np.ndarray, filepath: str):
        """Save depth map as numpy array."""
        np.save(filepath, depth)

    @staticmethod
    def load_depth_npy(filepath: str) -> np.ndarray:
        """Load depth map from numpy array."""
        return np.load(filepath)

    @staticmethod
    def visualize_depth(depth: np.ndarray) -> np.ndarray:
        """
        Create color visualization of depth map.

        Args:
            depth: Depth map

        Returns:
            RGB image with color-coded depth
        """
        # Normalize
        depth_normalized = depth - np.min(depth[depth > 0])
        depth_normalized = depth_normalized / np.max(depth_normalized)

        # Apply colormap
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return depth_colored


# Convenience function to create pattern generators
def create_stripe_pattern(resolution: Tuple[int, int],
                          frequency: float = 15.0,
                          orientation: str = 'vertical',
                          pattern_type: str = 'sinusoidal') -> np.ndarray:
    """
    Create a stripe pattern.

    Args:
        resolution: Pattern resolution (width, height)
        frequency: Stripe frequency
        orientation: 'vertical' or 'horizontal'
        pattern_type: 'sinusoidal' or 'binary'

    Returns:
        Pattern array (H x W) with values in [0, 1]
    """
    from structured_light_3d import StructuredLightProjector

    # Create temporary projector just for pattern generation
    temp_projector = StructuredLightProjector(
        position=np.array([0, 0, 0]),
        look_at=np.array([0, 0, 1]),
        fov=50.0,
        resolution=resolution
    )

    return temp_projector.create_stripe_pattern(frequency, orientation, pattern_type)


def create_dot_pattern(resolution: Tuple[int, int],
                      dot_spacing: int = 40,
                      dot_size: int = 3) -> np.ndarray:
    """Create a dot pattern."""
    from structured_light_3d import StructuredLightProjector

    temp_projector = StructuredLightProjector(
        position=np.array([0, 0, 0]),
        look_at=np.array([0, 0, 1]),
        fov=50.0,
        resolution=resolution
    )

    return temp_projector.create_dot_pattern(dot_spacing, dot_size)


def create_grid_pattern(resolution: Tuple[int, int],
                       grid_spacing: int = 60,
                       line_thickness: int = 3) -> np.ndarray:
    """Create a grid pattern."""
    from structured_light_3d import StructuredLightProjector

    temp_projector = StructuredLightProjector(
        position=np.array([0, 0, 0]),
        look_at=np.array([0, 0, 1]),
        fov=50.0,
        resolution=resolution
    )

    return temp_projector.create_grid_pattern(grid_spacing, line_thickness)
