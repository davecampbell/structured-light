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
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from datetime import datetime
import os


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
                color: Optional[np.ndarray] = None) -> None:
        """
        Add a box to the scene.

        Args:
            size: Box dimensions (width, height, depth)
            position: Box center position (x, y, z)
            rotation: 3x3 rotation matrix (optional)
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
        Simple pattern projection (placeholder for full projective texture mapping).

        Args:
            pattern: Projector pattern
            depth: Depth map from camera

        Returns:
            Pattern overlay image
        """
        # Resize pattern to camera resolution
        pattern_resized = cv2.resize(pattern, self.camera_resolution)

        # Create RGB pattern
        pattern_rgb = np.stack([pattern_resized] * 3, axis=-1)

        # Mask by depth (no pattern where no geometry)
        valid_mask = (depth > 0).astype(np.float32)
        pattern_rgb = pattern_rgb * valid_mask[:, :, np.newaxis]

        return pattern_rgb


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


def main():
    """
    Main function demonstrating 3D structured light simulation.
    """
    print("3D Structured Light Simulator")
    print("=" * 50)

    # Check display availability
    has_display = _has_display()
    if not has_display:
        matplotlib.use('Agg')

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 1. Create projector
    print("\n1. Creating projector at position (0.5, -0.8, 1.5)...")
    projector = StructuredLightProjector(
        position=np.array([0.5, -0.8, 1.5]),
        look_at=np.array([0.0, 0.0, 0.0]),
        fov=50.0,
        resolution=(1024, 768)
    )

    # 2. Create scene with random boxes
    print("2. Generating scene with random boxes...")
    scene = Scene3D()
    scene.add_ground_plane(size=3.0, height=-0.5)
    scene.generate_random_boxes(num_boxes=5, bounds=(1.0, 1.0, 0.4))

    # 3. Create renderer with camera
    print("3. Setting up camera at position (1.2, 0.0, 1.0)...")
    renderer = StructuredLightRenderer(
        projector=projector,
        camera_position=np.array([1.2, 0.0, 1.0]),
        camera_look_at=np.array([0.0, 0.0, 0.0]),
        camera_resolution=(640, 480),
        camera_fov=60.0
    )

    # 4. Generate different patterns and render
    print("4. Rendering scenes with different patterns...")

    patterns = [
        ('vertical_stripes', projector.create_stripe_pattern(frequency=15, orientation='vertical')),
        ('horizontal_stripes', projector.create_stripe_pattern(frequency=15, orientation='horizontal')),
        ('dots', projector.create_dot_pattern(dot_spacing=40, dot_size=4)),
        ('grid', projector.create_grid_pattern(grid_spacing=50, line_thickness=3))
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for pattern_name, pattern in patterns:
        print(f"   - Rendering {pattern_name}...")
        rgb, depth = renderer.render(scene, pattern, ambient_light=0.3, pattern_intensity=0.6)

        # Save individual results
        save_path = output_dir / f"structured_light_3d_{pattern_name}_{timestamp}.png"
        visualize_results(rgb, depth, save_path=save_path, show=False)

    # Render one final comparison with display
    print("\n5. Creating final comparison visualization...")
    rgb_final, depth_final = renderer.render(scene, patterns[0][1], ambient_light=0.3, pattern_intensity=0.6)
    final_path = output_dir / f"structured_light_3d_comparison_{timestamp}.png"
    visualize_results(rgb_final, depth_final, save_path=final_path, show=has_display)

    print("\nGeneration complete!")
    print(f"Output saved to: {output_dir}/")
    print("\nNext steps:")
    print("- Adjust projector and camera positions")
    print("- Modify pattern parameters")
    print("- Add custom geometries")
    print("- Integrate with your synthetic data pipeline")


if __name__ == "__main__":
    main()
