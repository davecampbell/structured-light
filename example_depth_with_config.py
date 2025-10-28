"""
Example: Using Config File with Depth-Based Approach

Demonstrates loading camera and projector calibrations from the same
YAML config file used by structured_light_3d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from structured_light_from_depth import (
    load_calibrations_from_config,
    DepthMapGenerator,
    PatternProjector,
    DepthMapIO,
    create_stripe_pattern,
    create_dot_pattern,
    create_grid_pattern
)
from structured_light_3d import Scene3D


def main():
    print("=" * 60)
    print("Depth-Based Projection Using Config File")
    print("=" * 60)
    print()

    # Load calibrations from config file
    print("Loading calibrations from config_example.yaml...")
    camera_calib, projector_calib = load_calibrations_from_config('config_example.yaml')

    print(f"✓ Camera: position={camera_calib.position}, fov={camera_calib.fov}°")
    print(f"✓ Projector: position={projector_calib.position}, fov={projector_calib.fov}°")
    print()

    # Create scene (for synthetic depth)
    print("Creating 3D scene...")
    scene = Scene3D()
    scene.add_ground_plane()
    scene.add_box(size=(0.3, 0.3, 0.3), position=np.array([0, 0, 0.15]))
    print("✓ Scene created")
    print()

    # Generate depth map
    print("Generating depth map from scene...")
    depth_generator = DepthMapGenerator(camera_calib)
    rgb_base, depth_map = depth_generator.render_depth(scene)
    print(f"✓ Depth map generated: {depth_map.shape}")
    print(f"  Depth range: {depth_map[depth_map > 0].min():.3f} - {depth_map.max():.3f} meters")
    print()

    # Create pattern projector
    print("Creating pattern projector...")
    pattern_projector = PatternProjector(camera_calib, projector_calib)
    print("✓ Pattern projector ready")
    print()

    # Project different patterns
    print("Projecting patterns...")
    results = []
    titles = []

    # Vertical stripes
    pattern = create_stripe_pattern(projector_calib.resolution, frequency=15, orientation='vertical')
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Vertical Stripes')
    print("  ✓ Vertical stripes")

    # Horizontal stripes
    pattern = create_stripe_pattern(projector_calib.resolution, frequency=15, orientation='horizontal')
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Horizontal Stripes')
    print("  ✓ Horizontal stripes")

    # Dots
    pattern = create_dot_pattern(projector_calib.resolution, dot_spacing=40, dot_size=3)
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Dots')
    print("  ✓ Dots")

    # Grid
    pattern = create_grid_pattern(projector_calib.resolution, grid_spacing=50, line_thickness=2)
    result = pattern_projector.project_pattern_backward(pattern, depth_map, rgb_base)
    results.append(result)
    titles.append('Grid')
    print("  ✓ Grid")
    print()

    # Visualize
    print("Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Depth-Based Projection (Config File)', fontsize=14, fontweight='bold')

    axes = axes.flatten()
    for i, (result, title) in enumerate(zip(results, titles)):
        axes[i].imshow(result)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('output/depth_config_patterns.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: output/depth_config_patterns.png")
    print()

    # Save depth map for later use
    print("Saving depth map...")
    DepthMapIO.save_depth_png(depth_map, 'output/config_depth.png', max_depth=5.0)
    print("✓ Saved: output/config_depth.png")
    print()

    print("=" * 60)
    print("Success!")
    print("=" * 60)
    print()
    print("Key advantage: Same config file works for both:")
    print("  • structured_light_3d.py (geometry-based)")
    print("  • structured_light_from_depth.py (depth-based)")
    print()
    print("This makes it easy to:")
    print("  ✓ Switch between approaches")
    print("  ✓ Share configurations")
    print("  ✓ Test different camera/projector setups")


if __name__ == '__main__':
    main()
