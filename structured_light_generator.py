"""
Synthetic Structured Light Generator

This script generates synthetic structured light patterns overlaid on images with depth data.
Used for data augmentation to improve vision models for robotic grasping tasks.

Requirements:
    pip install numpy opencv-python matplotlib pillow

Usage:
    python structured_light_generator.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional


class StructuredLightGenerator:
    """Generate various types of structured light patterns based on depth maps."""

    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        """
        Initialize the structured light generator.

        Args:
            image_size: (width, height) of the output images
        """
        self.width, self.height = image_size

    def create_stripe_pattern(self,
                            depth_map: np.ndarray,
                            frequency: float = 20.0,
                            orientation: str = 'vertical',
                            pattern_type: str = 'sinusoidal') -> np.ndarray:
        """
        Create a stripe/fringe pattern deformed by depth.

        Args:
            depth_map: 2D array of depth values (normalized 0-1 or actual depths)
            frequency: Number of stripes across the image
            orientation: 'vertical' or 'horizontal'
            pattern_type: 'sinusoidal' or 'binary'

        Returns:
            Structured light pattern as float array [0, 1]
        """
        h, w = depth_map.shape

        # Create coordinate grids
        x = np.linspace(0, 2 * np.pi * frequency, w)
        y = np.linspace(0, 2 * np.pi * frequency, h)

        if orientation == 'vertical':
            # Vertical stripes
            base_pattern = np.tile(x, (h, 1))
        else:
            # Horizontal stripes
            base_pattern = np.tile(y[:, np.newaxis], (1, w))

        # Modulate pattern with depth (depth creates phase shift)
        # Normalize depth to reasonable range for displacement
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        phase_shift = depth_normalized * np.pi * 0.5  # Depth affects phase

        modulated_pattern = base_pattern + phase_shift

        if pattern_type == 'sinusoidal':
            # Sinusoidal fringes
            pattern = (np.sin(modulated_pattern) + 1.0) / 2.0
        else:
            # Binary stripes
            pattern = (np.sin(modulated_pattern) > 0).astype(np.float32)

        return pattern

    def create_dot_pattern(self,
                          depth_map: np.ndarray,
                          dot_spacing: int = 30,
                          dot_size: int = 3) -> np.ndarray:
        """
        Create an IR-style dot pattern (like Kinect) with depth-based displacement.

        Args:
            depth_map: 2D array of depth values
            dot_spacing: Spacing between dots in pixels
            dot_size: Radius of each dot

        Returns:
            Dot pattern as float array [0, 1]
        """
        h, w = depth_map.shape
        pattern = np.zeros((h, w), dtype=np.float32)

        # Normalize depth for displacement
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # Create regular grid of dots
        for y in range(0, h, dot_spacing):
            for x in range(0, w, dot_spacing):
                # Displace dot based on local depth
                if y < h and x < w:
                    displacement = int(depth_normalized[y, x] * 15)  # Max 15 pixel shift
                    dot_x = min(w - 1, x + displacement)
                    dot_y = y

                    # Draw dot
                    cv2.circle(pattern, (dot_x, dot_y), dot_size, 1.0, -1)

        return pattern

    def create_grid_pattern(self,
                           depth_map: np.ndarray,
                           grid_spacing: int = 40,
                           line_thickness: int = 2) -> np.ndarray:
        """
        Create a deformed grid pattern based on depth.

        Args:
            depth_map: 2D array of depth values
            grid_spacing: Spacing between grid lines
            line_thickness: Thickness of grid lines

        Returns:
            Grid pattern as float array [0, 1]
        """
        h, w = depth_map.shape
        pattern = np.zeros((h, w), dtype=np.float32)

        # Normalize depth
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

        # Vertical lines
        for x in range(0, w, grid_spacing):
            for y in range(h):
                displacement = int(depth_normalized[y, x] * 20)
                new_x = min(w - 1, max(0, x + displacement))
                pattern[y, max(0, new_x-line_thickness//2):min(w, new_x+line_thickness//2)] = 1.0

        # Horizontal lines
        for y in range(0, h, grid_spacing):
            for x in range(w):
                if y < h:
                    displacement = int(depth_normalized[y, x] * 20)
                    new_y = min(h - 1, max(0, y + displacement))
                    pattern[max(0, new_y-line_thickness//2):min(h, new_y+line_thickness//2), x] = 1.0

        return pattern

    def add_realistic_noise(self,
                           pattern: np.ndarray,
                           noise_level: float = 0.05,
                           missing_regions: bool = True) -> np.ndarray:
        """
        Add realistic artifacts to structured light patterns.

        Args:
            pattern: Clean pattern
            noise_level: Amount of Gaussian noise (0-1)
            missing_regions: Whether to add missing data regions

        Returns:
            Noisy pattern
        """
        noisy_pattern = pattern.copy()

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, pattern.shape)
        noisy_pattern = np.clip(noisy_pattern + noise, 0, 1)

        # Add missing regions (like IR shadow regions)
        if missing_regions:
            # Create Perlin-like noise mask for missing regions
            mask = np.random.random(pattern.shape) > 0.95
            # Dilate the mask slightly
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            noisy_pattern[mask > 0] = 0

        return noisy_pattern

    def overlay_on_rgb(self,
                      rgb_image: np.ndarray,
                      pattern: np.ndarray,
                      blend_factor: float = 0.3) -> np.ndarray:
        """
        Overlay structured light pattern on RGB image.

        Args:
            rgb_image: RGB image (H, W, 3) in range [0, 255] or [0, 1]
            pattern: Structured light pattern (H, W) in range [0, 1]
            blend_factor: How much to blend (0 = no pattern, 1 = full pattern)

        Returns:
            Blended image
        """
        # Ensure RGB is float [0, 1]
        if rgb_image.max() > 1.0:
            rgb_image = rgb_image.astype(np.float32) / 255.0

        # Expand pattern to 3 channels
        pattern_rgb = np.stack([pattern] * 3, axis=-1)

        # Blend with additive mode (structured light adds to scene)
        blended = rgb_image + pattern_rgb * blend_factor
        blended = np.clip(blended, 0, 1)

        return blended


def create_synthetic_depth_map(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic depth map for testing (simulates boxes at different depths).

    Returns:
        Depth map with random box-like structures
    """
    depth_map = np.ones((height, width), dtype=np.float32) * 0.5  # Background depth

    # Add 3-5 random boxes at different depths
    num_boxes = np.random.randint(3, 6)
    for _ in range(num_boxes):
        x1 = np.random.randint(0, width - 100)
        y1 = np.random.randint(0, height - 100)
        box_w = np.random.randint(50, 150)
        box_h = np.random.randint(50, 150)
        x2 = min(width, x1 + box_w)
        y2 = min(height, y1 + box_h)

        depth = np.random.uniform(0.2, 0.8)
        depth_map[y1:y2, x1:x2] = depth

    # Add some Gaussian blur to make it more realistic
    depth_map = cv2.GaussianBlur(depth_map, (15, 15), 0)

    return depth_map


def visualize_all_patterns(generator: StructuredLightGenerator,
                          depth_map: np.ndarray,
                          rgb_image: Optional[np.ndarray] = None):
    """
    Generate and visualize all pattern types.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Synthetic Structured Light Patterns', fontsize=16)

    # Row 1: Stripe patterns
    vertical_stripes = generator.create_stripe_pattern(depth_map, frequency=15, orientation='vertical')
    horizontal_stripes = generator.create_stripe_pattern(depth_map, frequency=15, orientation='horizontal')
    binary_stripes = generator.create_stripe_pattern(depth_map, frequency=10, pattern_type='binary')

    axes[0, 0].imshow(vertical_stripes, cmap='gray')
    axes[0, 0].set_title('Vertical Sinusoidal Stripes')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(horizontal_stripes, cmap='gray')
    axes[0, 1].set_title('Horizontal Sinusoidal Stripes')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(binary_stripes, cmap='gray')
    axes[0, 2].set_title('Binary Stripes')
    axes[0, 2].axis('off')

    # Row 2: Dot and Grid patterns
    dot_pattern = generator.create_dot_pattern(depth_map, dot_spacing=25, dot_size=2)
    grid_pattern = generator.create_grid_pattern(depth_map, grid_spacing=35)
    noisy_stripes = generator.add_realistic_noise(vertical_stripes, noise_level=0.05)

    axes[1, 0].imshow(dot_pattern, cmap='gray')
    axes[1, 0].set_title('IR Dot Pattern')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(grid_pattern, cmap='gray')
    axes[1, 1].set_title('Deformed Grid')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(noisy_stripes, cmap='gray')
    axes[1, 2].set_title('Noisy Pattern with Missing Regions')
    axes[1, 2].axis('off')

    # Row 3: Depth map and overlays
    axes[2, 0].imshow(depth_map, cmap='viridis')
    axes[2, 0].set_title('Input Depth Map')
    axes[2, 0].axis('off')

    if rgb_image is not None:
        # Overlay on RGB
        overlay1 = generator.overlay_on_rgb(rgb_image, vertical_stripes, blend_factor=0.4)
        overlay2 = generator.overlay_on_rgb(rgb_image, dot_pattern, blend_factor=0.6)

        axes[2, 1].imshow(overlay1)
        axes[2, 1].set_title('RGB + Stripe Pattern')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(overlay2)
        axes[2, 2].set_title('RGB + Dot Pattern')
        axes[2, 2].axis('off')
    else:
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate structured light generation.
    """
    print("Synthetic Structured Light Generator")
    print("=" * 50)

    # Initialize generator
    generator = StructuredLightGenerator(image_size=(640, 480))

    # Create synthetic depth map for testing
    print("\n1. Creating synthetic depth map...")
    depth_map = create_synthetic_depth_map(640, 480)

    # Create a simple RGB image for overlay testing
    print("2. Creating synthetic RGB image...")
    rgb_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    # Generate and visualize all patterns
    print("3. Generating structured light patterns...")
    visualize_all_patterns(generator, depth_map, rgb_image)

    print("\nGeneration complete!")
    print("\nNext steps:")
    print("- Load your actual depth maps and RGB images")
    print("- Experiment with different pattern parameters")
    print("- Save patterns as additional training channels")
    print("- Integrate into your data pipeline")


if __name__ == "__main__":
    main()
