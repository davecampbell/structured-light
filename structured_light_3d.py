"""
3D Structured Light Simulator

This script simulates structured light projection with proper 3D geometry:
- Projector at specified (x, y, z) position casts patterns
- Camera at different (x, y, z) observes the scene
- Patterns projected onto 3D objects (boxes, planes) with realistic occlusions
- Full perspective rendering from camera viewpoint

Requirements:
    pip install numpy opencv-python matplotlib trimesh pyrender pillow

Usage:
    python structured_light_3d.py
"""

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import trimesh
import pyrender
import yaml
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime
import os
import sys


def _has_display() -> bool:
    """
    Check if a display is available for showing plots.

    Returns:
        True if display is available, False otherwise
    """
    if os.environ.get('DISPLAY'):
        return True
    if os.name == 'posix' and os.uname().sysname == 'Darwin':
        return True
    try:
        import tkinter
        tkinter.Tk().withdraw()
        return True
    except:
        return False


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (in degrees) to a 3x3 rotation matrix.
    Uses ZYX convention (yaw-pitch-roll).

    Args:
        roll: Rotation around X-axis in degrees
        pitch: Rotation around Y-axis in degrees
        yaw: Rotation around Z-axis in degrees

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    # Rotation around Y (pitch)
    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    # Rotation around Z (yaw)
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


class StructuredLightProjector:
    """Represents a structured light projector with position and pattern generation."""

    def __init__(self,
                 position: np.ndarray = np.array([0.0, 0.0, 2.0]),
                 look_at: np.ndarray = np.array([0.0, 0.0, 0.0]),
                 fov: float = 60.0,
                 resolution: Tuple[int, int] = (1024, 768)):
        """
        Initialize projector.

        Args:
            position: 3D position of projector (x, y, z)
            look_at: Point the projector is aimed at
            fov: Field of view in degrees
            resolution: Projector resolution (width, height)
        """
        self.position = position
        self.look_at = look_at
        self.fov = fov
        self.resolution = resolution

    def create_stripe_pattern(self,
                             frequency: float = 20.0,
                             orientation: str = 'vertical',
                             pattern_type: str = 'sinusoidal') -> np.ndarray:
        """
        Create a stripe pattern to project.

        Args:
            frequency: Number of stripes across the image
            orientation: 'vertical' or 'horizontal'
            pattern_type: 'sinusoidal' or 'binary'

        Returns:
            Pattern image as float array [0, 1]
        """
        h, w = self.resolution[1], self.resolution[0]

        x = np.linspace(0, 2 * np.pi * frequency, w)
        y = np.linspace(0, 2 * np.pi * frequency, h)

        if orientation == 'vertical':
            pattern = np.tile(x, (h, 1))
        else:
            pattern = np.tile(y[:, np.newaxis], (1, w))

        if pattern_type == 'sinusoidal':
            pattern = (np.sin(pattern) + 1.0) / 2.0
        else:
            pattern = (np.sin(pattern) > 0).astype(np.float32)

        return pattern

    def create_dot_pattern(self,
                          dot_spacing: int = 50,
                          dot_size: int = 5) -> np.ndarray:
        """
        Create an IR-style dot pattern.

        Args:
            dot_spacing: Spacing between dots in pixels
            dot_size: Radius of each dot

        Returns:
            Dot pattern as float array [0, 1]
        """
        h, w = self.resolution[1], self.resolution[0]
        pattern = np.zeros((h, w), dtype=np.float32)

        for y in range(dot_spacing // 2, h, dot_spacing):
            for x in range(dot_spacing // 2, w, dot_spacing):
                cv2.circle(pattern, (x, y), dot_size, 1.0, -1)

        return pattern

    def create_grid_pattern(self,
                           grid_spacing: int = 60,
                           line_thickness: int = 3) -> np.ndarray:
        """
        Create a grid pattern.

        Args:
            grid_spacing: Spacing between grid lines
            line_thickness: Thickness of grid lines

        Returns:
            Grid pattern as float array [0, 1]
        """
        h, w = self.resolution[1], self.resolution[0]
        pattern = np.zeros((h, w), dtype=np.float32)

        # Vertical lines
        for x in range(0, w, grid_spacing):
            pattern[:, max(0, x-line_thickness//2):min(w, x+line_thickness//2)] = 1.0

        # Horizontal lines
        for y in range(0, h, grid_spacing):
            pattern[max(0, y-line_thickness//2):min(h, y+line_thickness//2), :] = 1.0

        return pattern


class Scene3D:
    """Represents a 3D scene with objects."""

    def __init__(self):
        """Initialize empty scene."""
        self.meshes = []
        self.mesh_poses = []

    def add_box(self,
                size: Tuple[float, float, float] = (0.3, 0.3, 0.3),
                position: np.ndarray = np.array([0.0, 0.0, 0.0]),
                rotation: Optional[np.ndarray] = None,
                euler: Optional[Tuple[float, float, float]] = None,
                color: Optional[np.ndarray] = None) -> None:
        """
        Add a box to the scene.

        Args:
            size: Box dimensions (width, depth, height)
            position: Box center position (x, y, z)
            rotation: 3x3 rotation matrix (optional, overridden by euler if both provided)
            euler: Euler angles (roll, pitch, yaw) in degrees (optional)
            color: RGB color [0-255] (optional, random if None)
        """
        # Create box mesh
        box = trimesh.creation.box(extents=size)

        # Set color
        if color is None:
            color = np.random.randint(80, 180, 3)
        box.visual.vertex_colors = np.hstack([
            np.tile(color, (len(box.vertices), 1)),
            np.full((len(box.vertices), 1), 255)
        ])

        # Create pose matrix
        pose = np.eye(4)

        # Handle rotation - euler takes precedence if both provided
        if euler is not None:
            rotation = euler_to_rotation_matrix(*euler)

        if rotation is not None:
            pose[:3, :3] = rotation

        pose[:3, 3] = position

        self.meshes.append(box)
        self.mesh_poses.append(pose)

    def add_ground_plane(self,
                        size: float = 2.0,
                        height: float = -0.5,
                        color: np.ndarray = np.array([200, 200, 200])) -> None:
        """
        Add a ground plane to the scene.

        Args:
            size: Size of the plane
            height: Z-coordinate of the plane
            color: RGB color [0-255]
        """
        # Create plane mesh
        plane = trimesh.creation.box(extents=(size, size, 0.01))
        plane.visual.vertex_colors = np.hstack([
            np.tile(color, (len(plane.vertices), 1)),
            np.full((len(plane.vertices), 1), 255)
        ])

        # Position at specified height
        pose = np.eye(4)
        pose[2, 3] = height

        self.meshes.append(plane)
        self.mesh_poses.append(pose)

    def generate_random_boxes(self,
                            num_boxes: int = 5,
                            bounds: Tuple[float, float, float] = (0.8, 0.8, 0.3)) -> None:
        """
        Generate random boxes in the scene.

        Args:
            num_boxes: Number of boxes to generate
            bounds: (x_range, y_range, z_range) for random placement
        """
        for _ in range(num_boxes):
            # Random size
            size = np.random.uniform(0.1, 0.3, 3)

            # Random position within bounds
            position = np.array([
                np.random.uniform(-bounds[0]/2, bounds[0]/2),
                np.random.uniform(-bounds[1]/2, bounds[1]/2),
                np.random.uniform(0, bounds[2])
            ])

            # Random rotation
            angles = np.random.uniform(0, 2*np.pi, 3)
            rx = trimesh.transformations.rotation_matrix(angles[0], [1, 0, 0])[:3, :3]
            ry = trimesh.transformations.rotation_matrix(angles[1], [0, 1, 0])[:3, :3]
            rz = trimesh.transformations.rotation_matrix(angles[2], [0, 0, 1])[:3, :3]
            rotation = rz @ ry @ rx

            self.add_box(size=size, position=position, rotation=rotation)


class StructuredLightRenderer:
    """Renders structured light scenes with projector and camera."""

    def __init__(self,
                 projector: StructuredLightProjector,
                 camera_position: np.ndarray = np.array([1.0, 0.0, 1.5]),
                 camera_look_at: np.ndarray = np.array([0.0, 0.0, 0.0]),
                 camera_resolution: Tuple[int, int] = (640, 480),
                 camera_fov: float = 60.0):
        """
        Initialize renderer.

        Args:
            projector: StructuredLightProjector instance
            camera_position: Camera 3D position
            camera_look_at: Point camera is aimed at
            camera_resolution: Camera resolution (width, height)
            camera_fov: Camera field of view in degrees
        """
        self.projector = projector
        self.camera_position = camera_position
        self.camera_look_at = camera_look_at
        self.camera_resolution = camera_resolution
        self.camera_fov = camera_fov

    def render(self,
               scene: Scene3D,
               pattern: np.ndarray,
               ambient_light: float = 0.3,
               pattern_intensity: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render the scene with structured light.

        Args:
            scene: Scene3D object with meshes
            pattern: Pattern to project from projector
            ambient_light: Ambient light intensity [0, 1]
            pattern_intensity: Projected pattern intensity [0, 1]

        Returns:
            Tuple of (rgb_image, depth_map)
        """
        # Create pyrender scene
        pr_scene = pyrender.Scene(ambient_light=[ambient_light] * 3)

        # Add meshes to scene
        for mesh, pose in zip(scene.meshes, scene.mesh_poses):
            pr_mesh = pyrender.Mesh.from_trimesh(mesh)
            pr_scene.add(pr_mesh, pose=pose)

        # Create camera
        camera = pyrender.PerspectiveCamera(
            yfov=np.radians(self.camera_fov),
            aspectRatio=self.camera_resolution[0] / self.camera_resolution[1]
        )

        # Camera pose matrix
        camera_pose = self._look_at_matrix(self.camera_position, self.camera_look_at)
        pr_scene.add(camera, pose=camera_pose)

        # Add projector as spotlight with pattern texture
        # For now, we'll render base scene and add pattern in post-processing
        # Full texture projection requires more complex shader setup

        # Add directional light from projector direction
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_pose = self._look_at_matrix(self.projector.position, self.projector.look_at)
        pr_scene.add(light, pose=light_pose)

        # Add additional light for better visibility
        extra_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        pr_scene.add(extra_light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.camera_resolution[0],
            viewport_height=self.camera_resolution[1]
        )

        color, depth = renderer.render(pr_scene)
        renderer.delete()

        # Project pattern onto scene (simplified approach)
        # In a full implementation, this would use projective texture mapping
        pattern_overlay = self._project_pattern_simple(pattern, depth)

        # Combine base rendering with pattern
        color_float = color.astype(np.float32) / 255.0
        result = color_float + pattern_overlay * pattern_intensity
        result = np.clip(result, 0, 1)
        result = (result * 255).astype(np.uint8)

        return result, depth

    def _look_at_matrix(self, eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Create a look-at transformation matrix (OpenGL convention).

        Args:
            eye: Camera/light position
            target: Point to look at

        Returns:
            4x4 transformation matrix
        """
        # Camera looks down -Z axis in OpenGL
        forward = eye - target  # Note: opposite direction for camera
        forward = forward / np.linalg.norm(forward)

        # Use world up vector
        world_up = np.array([0.0, 0.0, 1.0])

        # Compute right vector
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            # Forward is parallel to up, use alternate up
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

    def _project_pattern_simple(self,
                                pattern: np.ndarray,
                                depth: np.ndarray) -> np.ndarray:
        """
        Project pattern onto scene using projective texture mapping.

        Args:
            pattern: Projector pattern
            depth: Depth map from camera

        Returns:
            Pattern overlay image
        """
        h, w = self.camera_resolution[1], self.camera_resolution[0]
        pattern_rgb = np.zeros((h, w, 3), dtype=np.float32)

        # Create camera intrinsics matrix (simplified)
        camera_fov_rad = np.radians(self.camera_fov)
        camera_fx = w / (2 * np.tan(camera_fov_rad / 2))
        camera_fy = h / (2 * np.tan(camera_fov_rad / 2))
        camera_cx = w / 2
        camera_cy = h / 2

        # Camera pose
        camera_pose = self._look_at_matrix(self.camera_position, self.camera_look_at)
        camera_pose_inv = np.linalg.inv(camera_pose)

        # Projector intrinsics (simplified)
        proj_fov_rad = np.radians(self.projector.fov)
        proj_w, proj_h = self.projector.resolution
        proj_fx = proj_w / (2 * np.tan(proj_fov_rad / 2))
        proj_fy = proj_h / (2 * np.tan(proj_fov_rad / 2))
        proj_cx = proj_w / 2
        proj_cy = proj_h / 2

        # Projector pose
        proj_pose = self._look_at_matrix(self.projector.position, self.projector.look_at)
        proj_pose_inv = np.linalg.inv(proj_pose)

        # For each camera pixel, project to 3D, then to projector space
        for v in range(h):
            for u in range(w):
                d = depth[v, u]
                if d <= 0:
                    continue

                # Back-project from camera to 3D world coordinates
                # Camera ray in camera space
                x_cam = (u - camera_cx) * d / camera_fx
                y_cam = (v - camera_cy) * d / camera_fy
                z_cam = d

                # Transform to world coordinates
                point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                point_world = camera_pose @ point_cam

                # Transform to projector coordinates
                point_proj = proj_pose_inv @ point_world

                # Project onto projector image plane
                if point_proj[2] > 0:  # In front of projector
                    proj_u = proj_fx * point_proj[0] / point_proj[2] + proj_cx
                    proj_v = proj_fy * point_proj[1] / point_proj[2] + proj_cy

                    # Bilinear interpolation for smoother sampling
                    if 0 <= proj_u < proj_w - 1 and 0 <= proj_v < proj_h - 1:
                        u0 = int(np.floor(proj_u))
                        v0 = int(np.floor(proj_v))
                        u1 = u0 + 1
                        v1 = v0 + 1

                        # Interpolation weights
                        wu = proj_u - u0
                        wv = proj_v - v0

                        # Bilinear interpolation
                        pattern_value = (
                            pattern[v0, u0] * (1 - wu) * (1 - wv) +
                            pattern[v0, u1] * wu * (1 - wv) +
                            pattern[v1, u0] * (1 - wu) * wv +
                            pattern[v1, u1] * wu * wv
                        )
                        pattern_rgb[v, u, :] = pattern_value

        return pattern_rgb


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_scene_from_config(config: Dict[str, Any]) -> Scene3D:
    """
    Build a 3D scene from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Scene3D object
    """
    scene = Scene3D()
    scene_config = config.get('scene', {})

    # Add ground plane if enabled
    if scene_config.get('ground_plane', {}).get('enabled', True):
        gp = scene_config['ground_plane']
        scene.add_ground_plane(
            size=gp.get('size', 2.0),
            height=gp.get('height', -0.5),
            color=np.array(gp.get('color', [200, 200, 200]))
        )

    # Add manually specified boxes
    for box_config in scene_config.get('boxes', []):
        scene.add_box(
            size=tuple(box_config['size']),
            position=np.array(box_config['position']),
            euler=tuple(box_config['orientation']) if 'orientation' in box_config else None,
            color=np.array(box_config['color']) if box_config.get('color') is not None else None
        )

    # Add random boxes if enabled
    if scene_config.get('random_boxes', {}).get('enabled', False):
        rb = scene_config['random_boxes']
        count = rb.get('count', 5)

        size_min = np.array(rb['size_range']['min'])
        size_max = np.array(rb['size_range']['max'])

        pos_bounds = rb['position_bounds']
        ori_range = rb['orientation_range']

        for _ in range(count):
            size = np.random.uniform(size_min, size_max)
            position = np.array([
                np.random.uniform(*pos_bounds['x']),
                np.random.uniform(*pos_bounds['y']),
                np.random.uniform(*pos_bounds['z'])
            ])
            euler = (
                np.random.uniform(*ori_range['roll']),
                np.random.uniform(*ori_range['pitch']),
                np.random.uniform(*ori_range['yaw'])
            )
            scene.add_box(size=tuple(size), position=position, euler=euler)

    return scene


def visualize_results(rgb_image: np.ndarray,
                     depth_map: np.ndarray,
                     save_path: Optional[Path] = None,
                     show: bool = True) -> None:
    """
    Visualize rendering results.

    Args:
        rgb_image: Rendered RGB image
        depth_map: Rendered depth map
        save_path: Path to save figure (optional)
        show: Whether to show the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('3D Structured Light Simulation', fontsize=16)

    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB with Structured Light')
    axes[0].axis('off')

    axes[1].imshow(depth_map, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_multiple_results(results: List[Tuple[str, np.ndarray, np.ndarray]],
                               save_path: Optional[Path] = None,
                               show: bool = True) -> None:
    """
    Visualize multiple rendering results in a grid.

    Args:
        results: List of (pattern_name, rgb_image, depth_map) tuples
        save_path: Path to save figure (optional)
        show: Whether to show the plot
    """
    num_patterns = len(results)
    fig, axes = plt.subplots(num_patterns, 2, figsize=(12, 5 * num_patterns))
    fig.suptitle('3D Structured Light Simulation - Multiple Patterns', fontsize=16)

    # Handle single pattern case
    if num_patterns == 1:
        axes = [axes]

    for idx, (pattern_name, rgb_image, depth_map) in enumerate(results):
        axes[idx][0].imshow(rgb_image)
        axes[idx][0].set_title(f'{pattern_name} - RGB with Structured Light')
        axes[idx][0].axis('off')

        axes[idx][1].imshow(depth_map, cmap='viridis')
        axes[idx][1].set_title(f'{pattern_name} - Depth Map')
        axes[idx][1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-pattern visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """
    Main function demonstrating 3D structured light simulation.
    Supports config file via command line argument.

    Usage:
        python structured_light_3d.py                    # Use defaults
        python structured_light_3d.py config.yaml        # Use config file
    """
    print("3D Structured Light Simulator")
    print("=" * 50)

    # Check for config file argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else None

    if config_path:
        print(f"\nLoading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        print("\nNo config file specified, using default parameters")
        print("Usage: python structured_light_3d.py [config.yaml]")
        config = None

    # Check display availability
    has_display = _has_display()
    if not has_display:
        matplotlib.use('Agg')

    # Setup output directory
    if config and 'output' in config:
        output_dir = Path(config['output'].get('directory', 'output'))
        output_prefix = config['output'].get('prefix', 'structured_light_3d')
    else:
        output_dir = Path("output")
        output_prefix = "structured_light_3d"

    output_dir.mkdir(exist_ok=True)

    # 1. Create projector
    if config and 'projector' in config:
        proj_cfg = config['projector']
        print(f"\n1. Creating projector at position {proj_cfg['position']}...")
        projector = StructuredLightProjector(
            position=np.array(proj_cfg['position']),
            look_at=np.array(proj_cfg['look_at']),
            fov=proj_cfg['fov'],
            resolution=tuple(proj_cfg['resolution'])
        )
    else:
        print("\n1. Creating projector at position (0.5, -0.8, 1.5)...")
        projector = StructuredLightProjector(
            position=np.array([0.5, -0.8, 1.5]),
            look_at=np.array([0.0, 0.0, 0.0]),
            fov=50.0,
            resolution=(1024, 768)
        )

    # 2. Create scene
    if config and 'scene' in config:
        print("2. Building scene from configuration...")
        scene = build_scene_from_config(config)
    else:
        print("2. Generating scene with random boxes...")
        scene = Scene3D()
        scene.add_ground_plane(size=3.0, height=-0.5)
        scene.generate_random_boxes(num_boxes=5, bounds=(1.0, 1.0, 0.4))

    # 3. Create renderer with camera
    if config and 'camera' in config:
        cam_cfg = config['camera']
        print(f"3. Setting up camera at position {cam_cfg['position']}...")
        renderer = StructuredLightRenderer(
            projector=projector,
            camera_position=np.array(cam_cfg['position']),
            camera_look_at=np.array(cam_cfg['look_at']),
            camera_resolution=tuple(cam_cfg['resolution']),
            camera_fov=cam_cfg['fov']
        )
    else:
        print("3. Setting up camera at position (1.2, 0.0, 1.0)...")
        renderer = StructuredLightRenderer(
            projector=projector,
            camera_position=np.array([1.2, 0.0, 1.0]),
            camera_look_at=np.array([0.0, 0.0, 0.0]),
            camera_resolution=(640, 480),
            camera_fov=60.0
        )

    # Get rendering parameters
    if config and 'rendering' in config:
        ambient_light = config['rendering'].get('ambient_light', 0.3)
        pattern_intensity = config['rendering'].get('pattern_intensity', 0.6)
    else:
        ambient_light = 0.3
        pattern_intensity = 0.6

    # 4. Generate patterns
    print("4. Generating patterns and rendering...")

    display_pattern_indices = []  # Track which patterns to display
    if config and 'patterns' in config:
        patterns = []
        for idx, pat_cfg in enumerate(config['patterns']):
            pat_type = pat_cfg['type']

            # Check if this pattern should be displayed
            if pat_cfg.get('display', False):
                display_pattern_indices.append(idx)

            if 'stripes' in pat_type:
                orientation = 'vertical' if 'vertical' in pat_type else 'horizontal'
                pattern = projector.create_stripe_pattern(
                    frequency=pat_cfg.get('frequency', 15.0),
                    orientation=orientation,
                    pattern_type=pat_cfg.get('pattern_type', 'sinusoidal')
                )
                patterns.append((pat_type, pattern))
            elif pat_type == 'dots':
                pattern = projector.create_dot_pattern(
                    dot_spacing=pat_cfg.get('dot_spacing', 40),
                    dot_size=pat_cfg.get('dot_size', 4)
                )
                patterns.append((pat_type, pattern))
            elif pat_type == 'grid':
                pattern = projector.create_grid_pattern(
                    grid_spacing=pat_cfg.get('grid_spacing', 50),
                    line_thickness=pat_cfg.get('line_thickness', 3)
                )
                patterns.append((pat_type, pattern))
    else:
        patterns = [
            ('vertical_stripes', projector.create_stripe_pattern(frequency=15, orientation='vertical')),
            ('horizontal_stripes', projector.create_stripe_pattern(frequency=15, orientation='horizontal')),
            ('dots', projector.create_dot_pattern(dot_spacing=40, dot_size=4)),
            ('grid', projector.create_grid_pattern(grid_spacing=50, line_thickness=3))
        ]
        display_pattern_indices = [0]  # Default to first pattern

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine what to save
    save_individual = True
    save_comparison = True
    if config and 'output' in config:
        save_individual = config['output'].get('save_individual_patterns', True)
        save_comparison = config['output'].get('save_comparison', True)

    # Render patterns
    if save_individual:
        for pattern_name, pattern in patterns:
            print(f"   - Rendering {pattern_name}...")
            rgb, depth = renderer.render(scene, pattern, ambient_light=ambient_light, pattern_intensity=pattern_intensity)

            # Save individual results
            save_path = output_dir / f"{output_prefix}_{pattern_name}_{timestamp}.png"
            visualize_results(rgb, depth, save_path=save_path, show=False)

    # Render final comparison (with multiple patterns if multiple display flags set)
    if save_comparison:
        # Use patterns marked for display, or default to first pattern
        if not display_pattern_indices:
            display_pattern_indices = [0]

        # Render all patterns marked for display
        display_results = []
        pattern_names = []

        for idx in display_pattern_indices:
            pattern_name, pattern = patterns[idx]
            pattern_names.append(pattern_name)
            print(f"\n5. Rendering comparison for: {pattern_name}...")
            rgb, depth = renderer.render(scene, pattern, ambient_light=ambient_light, pattern_intensity=pattern_intensity)
            display_results.append((pattern_name, rgb, depth))

        # Save and show the comparison
        final_path = output_dir / f"{output_prefix}_comparison_{timestamp}.png"

        if len(display_results) == 1:
            # Single pattern - use simple visualization
            print(f"\nCreating comparison visualization (displaying: {pattern_names[0]})...")
            visualize_results(display_results[0][1], display_results[0][2], save_path=final_path, show=has_display)
        else:
            # Multiple patterns - use grid visualization
            print(f"\nCreating multi-pattern comparison visualization (displaying: {', '.join(pattern_names)})...")
            visualize_multiple_results(display_results, save_path=final_path, show=has_display)

    print("\nGeneration complete!")
    print(f"Output saved to: {output_dir}/")
    if not config_path:
        print("\nTip: Create a config file to customize all parameters!")
        print("Example: python structured_light_3d.py config_example.yaml")
    print("\nNext steps:")
    print("- Adjust projector and camera positions in config")
    print("- Modify box placements and orientations")
    print("- Add custom geometries")
    print("- Integrate with your synthetic data pipeline")


if __name__ == "__main__":
    main()
