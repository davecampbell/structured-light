"""
Example: Structured Light Projection from Depth Maps

This example demonstrates two workflows:
1. Generate synthetic depth from 3D geometry, then project patterns
2. Load external depth map and project patterns

This approach is useful when:
- You have depth data from a real depth camera
- You want to test with synthetic depth before using real data
- You want to decouple pattern projection from geometry knowledge
"""

import numpy as np
import matplotlib.pyplot as plt
from structured_light_from_depth import (
    CameraCalibration,
    ProjectorCalibration,
    DepthMapGenerator,
    PatternProjector,
    DepthMapIO,
    create_stripe_pattern,
    create_dot_pattern,
    create_grid_pattern
)
from structured_light_3d import Scene3D


def example_synthetic_depth():
    """Example: Generate depth from geometry, then project patterns."""

    print("=" * 60)
    print("Example 1: Synthetic Depth from Geometry")
    print("=" * 60)

    # Step 1: Define camera calibration
    camera_calib = CameraCalibration(
        position=np.array([1.2, 0.0, 1.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        fov=60.0,
        resolution=(640, 480)
    )

    # Step 2: Define projector calibration
    projector_calib = ProjectorCalibration(
        position=np.array([0.5, -0.8, 1.5]),
        look_at=np.array([0.0, 0.0, 0.0]),
        fov=50.0,
        resolution=(1024, 768)
    )

    # Step 3: Create 3D scene (only for depth generation)
    scene = Scene3D()
    scene.add_ground_plane()
    scene.add_box(size=(0.3, 0.3, 0.3), position=np.array([0, 0, 0.15]))

    # Step 4: Generate depth map from scene
    depth_generator = DepthMapGenerator(camera_calib)
    rgb_base, depth_map = depth_generator.render_depth(scene)

    print(f"✓ Generated depth map: {depth_map.shape}")
    print(f"  Depth range: {depth_map[depth_map > 0].min():.3f} - {depth_map.max():.3f} meters")

    # Step 5: Save depth map for later use
    DepthMapIO.save_depth_png(depth_map, 'output/synthetic_depth.png', max_depth=5.0)
    DepthMapIO.save_depth_npy(depth_map, 'output/synthetic_depth.npy')
    print("✓ Saved depth map: output/synthetic_depth.png, output/synthetic_depth.npy")

    # Step 6: Create pattern projector (no geometry needed!)
    pattern_projector = PatternProjector(camera_calib, projector_calib)

    # Step 7: Project different patterns
    results = []
    titles = []

    # Vertical stripes
    pattern = create_stripe_pattern(projector_calib.resolution, frequency=15, orientation='vertical')
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Vertical Stripes')

    # Horizontal stripes
    pattern = create_stripe_pattern(projector_calib.resolution, frequency=15, orientation='horizontal')
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Horizontal Stripes')

    # Dots
    pattern = create_dot_pattern(projector_calib.resolution, dot_spacing=40, dot_size=3)
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Dots')

    # Grid
    pattern = create_grid_pattern(projector_calib.resolution, grid_spacing=50, line_thickness=2)
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Grid')

    # Depth visualization
    depth_vis = DepthMapIO.visualize_depth(depth_map)
    results.append(depth_vis)
    titles.append('Depth Map (colored)')

    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Synthetic Depth → Structured Light Patterns', fontsize=14, fontweight='bold')

    axes = axes.flatten()
    for i, (result, title) in enumerate(zip(results, titles)):
        axes[i].imshow(result)
        axes[i].set_title(title)
        axes[i].axis('off')

    # Hide last subplot
    axes[5].imshow(rgb_base)
    axes[5].set_title('Base RGB')
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig('output/synthetic_depth_patterns.png', dpi=150, bbox_inches='tight')
    print("✓ Saved results: output/synthetic_depth_patterns.png")
    print()


def example_external_depth():
    """Example: Load external depth and project patterns."""

    print("=" * 60)
    print("Example 2: External Depth Map")
    print("=" * 60)

    # Step 1: Load depth map (using the synthetic one we saved as example)
    # In practice, this could be from a RealSense, Kinect, etc.
    depth_map = DepthMapIO.load_depth_png('output/synthetic_depth.png', max_depth=5.0)
    print(f"✓ Loaded depth map: {depth_map.shape}")
    print(f"  Depth range: {depth_map[depth_map > 0].min():.3f} - {depth_map.max():.3f} meters")

    # Step 2: Define camera calibration (must match the depth camera!)
    # These parameters would come from your depth camera calibration
    camera_calib = CameraCalibration(
        position=np.array([1.2, 0.0, 1.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        fov=60.0,
        resolution=(640, 480)
    )

    # Step 3: Define projector calibration
    projector_calib = ProjectorCalibration(
        position=np.array([0.5, -0.8, 1.5]),
        look_at=np.array([0.0, 0.0, 0.0]),
        fov=50.0,
        resolution=(1024, 768)
    )

    # Step 4: Create pattern projector
    pattern_projector = PatternProjector(camera_calib, projector_calib)

    # Step 5: Project pattern (no geometry needed!)
    pattern = create_stripe_pattern(projector_calib.resolution, frequency=20, orientation='vertical')
    result = pattern_projector.project_pattern_backward(pattern, depth_map)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('External Depth Map → Structured Light', fontsize=14, fontweight='bold')

    depth_vis = DepthMapIO.visualize_depth(depth_map)
    axes[0].imshow(depth_vis)
    axes[0].set_title('Input: Depth Map')
    axes[0].axis('off')

    axes[1].imshow(result)
    axes[1].set_title('Output: Structured Light Projection')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('output/external_depth_pattern.png', dpi=150, bbox_inches='tight')
    print("✓ Saved results: output/external_depth_pattern.png")
    print()


def example_depth_comparison():
    """Compare different depth sources."""

    print("=" * 60)
    print("Example 3: Depth Map Comparison")
    print("=" * 60)

    # This example shows how you could compare:
    # - Synthetic depth (ground truth)
    # - Real depth camera data (with noise)
    # - Filtered depth

    # Load synthetic depth
    depth_synthetic = DepthMapIO.load_depth_npy('output/synthetic_depth.npy')

    # Simulate noisy depth (like from a real sensor)
    noise = np.random.normal(0, 0.01, depth_synthetic.shape)
    depth_noisy = depth_synthetic + noise
    depth_noisy = np.clip(depth_noisy, 0, None)

    # Simple filtering
    import scipy.ndimage
    depth_filtered = scipy.ndimage.gaussian_filter(depth_noisy, sigma=1.0)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Depth Map Quality Comparison', fontsize=14, fontweight='bold')

    depths = [depth_synthetic, depth_noisy, depth_filtered]
    titles = ['Synthetic (Ground Truth)', 'Noisy (Simulated Sensor)', 'Filtered']

    for ax, depth, title in zip(axes, depths, titles):
        depth_vis = DepthMapIO.visualize_depth(depth)
        ax.imshow(depth_vis)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('output/depth_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison: output/depth_comparison.png")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Structured Light from Depth Maps - Examples")
    print("=" * 60)
    print()

    # Run examples
    example_synthetic_depth()
    example_external_depth()
    example_depth_comparison()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey advantages of this approach:")
    print("  ✓ Works with real depth cameras (RealSense, Kinect, etc.)")
    print("  ✓ Decoupled from 3D geometry knowledge")
    print("  ✓ Can save/load depth maps for reuse")
    print("  ✓ Easy to test synthetic vs. real data")
    print("  ✓ Depth-only processing (no mesh required)")
