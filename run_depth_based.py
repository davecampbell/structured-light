#!/usr/bin/env python3
"""
Run Depth-Based Structured Light Projection

Command line interface for the depth-based structured light system.
Loads configuration from YAML file and generates patterns using only depth maps.

Usage:
    python run_depth_based.py [config_file] [--depth DEPTH_FILE]
    python run_depth_based.py config_example.yaml
    python run_depth_based.py config_example.yaml --depth my_depth.npy
    python run_depth_based.py config_example.yaml --depth my_depth.png --max-depth 5.0
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from structured_light_from_depth import (
    load_calibrations_from_config,
    DepthMapGenerator,
    PatternProjector,
    DepthMapIO,
    create_stripe_pattern,
    create_dot_pattern,
    create_grid_pattern
)
from structured_light_3d import Scene3D, build_scene_from_config, load_config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Depth-Based Structured Light Projection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate depth from config geometry
  python run_depth_based.py config_example.yaml

  # Use external depth map (numpy)
  python run_depth_based.py config_example.yaml --depth my_depth.npy

  # Use external depth map (PNG)
  python run_depth_based.py config_example.yaml --depth my_depth.png --max-depth 5.0
        """
    )
    parser.add_argument('config', nargs='?', default='config_example.yaml',
                       help='YAML configuration file (default: config_example.yaml)')
    parser.add_argument('--depth', type=str,
                       help='External depth map file (.npy or .png)')
    parser.add_argument('--max-depth', type=float, default=5.0,
                       help='Max depth for PNG scaling (default: 5.0 meters)')

    args = parser.parse_args()
    config_path = args.config

    print("=" * 70)
    print("Depth-Based Structured Light Projection")
    print("=" * 70)
    print(f"\nLoading configuration from: {config_path}")

    # Check if config file exists
    if not Path(config_path).exists():
        print(f"\nError: Config file '{config_path}' not found!")
        parser.print_help()
        sys.exit(1)

    # Load full config (for scene and patterns)
    config = load_config(config_path)

    # Load calibrations
    camera_calib, projector_calib = load_calibrations_from_config(config_path)
    print(f"✓ Camera: position={camera_calib.position}, fov={camera_calib.fov}°, "
          f"resolution={camera_calib.resolution}")
    print(f"✓ Projector: position={projector_calib.position}, fov={projector_calib.fov}°, "
          f"resolution={projector_calib.resolution}")

    # Load or generate depth map
    if args.depth:
        # Load external depth map
        print(f"\nLoading external depth map from: {args.depth}")

        if not Path(args.depth).exists():
            print(f"Error: Depth file '{args.depth}' not found!")
            sys.exit(1)

        depth_path = Path(args.depth)
        if depth_path.suffix == '.npy':
            depth_map = DepthMapIO.load_depth_npy(str(depth_path))
            print(f"✓ Loaded numpy depth map: {depth_map.shape}")
        elif depth_path.suffix == '.png':
            depth_map = DepthMapIO.load_depth_png(str(depth_path), max_depth=args.max_depth)
            print(f"✓ Loaded PNG depth map: {depth_map.shape} (max_depth={args.max_depth}m)")
        else:
            print(f"Error: Unsupported depth format '{depth_path.suffix}'")
            print("Supported formats: .npy, .png")
            sys.exit(1)

        print(f"  Depth range: {depth_map[depth_map > 0].min():.3f} - {depth_map.max():.3f} meters")

        # Create a simple gray base image
        rgb_base = np.ones((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8) * 128

    else:
        # Generate depth from geometry
        print("\nBuilding 3D scene from config...")
        scene = build_scene_from_config(config)
        print(f"✓ Scene created with {len(scene.meshes)} meshes")

        print("\nGenerating depth map from geometry...")
        depth_generator = DepthMapGenerator(camera_calib)
        rgb_base, depth_map = depth_generator.render_depth(scene)
        print(f"✓ Depth map: {depth_map.shape}")
        print(f"  Depth range: {depth_map[depth_map > 0].min():.3f} - {depth_map.max():.3f} meters")

    # Save depth map (only if generated, not if loaded from external)
    output_dir = Path(config.get('output', {}).get('directory', 'output'))
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if not args.depth:
        depth_npy_path = output_dir / f'{timestamp}_depth_map.npy'
        depth_png_path = output_dir / f'{timestamp}_depth_map.png'
        DepthMapIO.save_depth_npy(depth_map, str(depth_npy_path))
        DepthMapIO.save_depth_png(depth_map, str(depth_png_path), max_depth=5.0)
        print(f"✓ Saved generated depth: {depth_npy_path}, {depth_png_path}")
    else:
        print(f"✓ Using external depth (not saving duplicate)")

    # Create pattern projector
    print("\nCreating pattern projector...")
    pattern_projector = PatternProjector(camera_calib, projector_calib)
    print("✓ Pattern projector ready")

    # Get pattern configs
    pattern_configs = config.get('patterns', [])
    if not pattern_configs:
        # Default patterns
        pattern_configs = [
            {'type': 'vertical_stripes', 'frequency': 15.0, 'display': True},
            {'type': 'horizontal_stripes', 'frequency': 15.0, 'display': True},
            {'type': 'dots', 'dot_spacing': 40, 'dot_size': 3, 'display': True},
            {'type': 'grid', 'grid_spacing': 50, 'line_thickness': 2, 'display': True}
        ]

    # Filter patterns by display flag
    patterns_to_generate = [p for p in pattern_configs if p.get('display', True)]

    if not patterns_to_generate:
        print("\nNo patterns enabled (all have display: false)")
        print("Set display: true in config to generate patterns")
        return

    # Project patterns
    print(f"\nProjecting {len(patterns_to_generate)} patterns (skipping {len(pattern_configs) - len(patterns_to_generate)} with display: false)...")
    results = []
    display_results = []
    display_titles = []

    for i, pattern_config in enumerate(patterns_to_generate):
        pattern_type = pattern_config['type']
        print(f"  [{i+1}/{len(patterns_to_generate)}] {pattern_type}...", end=' ')

        # Create pattern
        if 'stripe' in pattern_type:
            orientation = 'vertical' if 'vertical' in pattern_type else 'horizontal'
            frequency = pattern_config.get('frequency', 15.0)
            pattern_type_str = pattern_config.get('pattern_type', 'sinusoidal')
            pattern = create_stripe_pattern(
                projector_calib.resolution,
                frequency=frequency,
                orientation=orientation,
                pattern_type=pattern_type_str
            )
        elif pattern_type == 'dots':
            dot_spacing = pattern_config.get('dot_spacing', 40)
            dot_size = pattern_config.get('dot_size', 3)
            pattern = create_dot_pattern(
                projector_calib.resolution,
                dot_spacing=dot_spacing,
                dot_size=dot_size
            )
        elif pattern_type == 'grid':
            grid_spacing = pattern_config.get('grid_spacing', 60)
            line_thickness = pattern_config.get('line_thickness', 3)
            pattern = create_grid_pattern(
                projector_calib.resolution,
                grid_spacing=grid_spacing,
                line_thickness=line_thickness
            )
        else:
            print(f"Unknown pattern type: {pattern_type}")
            continue

        # Project pattern
        rendering_config = config.get('rendering', {})
        pattern_intensity = rendering_config.get('pattern_intensity', 0.6)
        result = pattern_projector.project_pattern_backward(
            pattern,
            depth_map,
            rgb_base,
            pattern_intensity=pattern_intensity
        )

        results.append((pattern_type, result))

        # Save individual pattern if configured
        if config.get('output', {}).get('save_individual_patterns', True):
            filename = f"{timestamp}_{pattern_type}.png"
            filepath = output_dir / filename
            plt.imsave(str(filepath), result)
            print(f"✓ Saved to {filepath}")
        else:
            print("✓")

        # Add to display (all patterns in this loop are already filtered for display: true)
        display_results.append(result)
        display_titles.append(pattern_type.replace('_', ' ').title())

    # Create comparison visualization if multiple patterns to display
    if len(display_results) > 1 and config.get('output', {}).get('save_comparison', True):
        print("\nCreating comparison visualization...")
        n_patterns = len(display_results)

        # Calculate grid layout
        if n_patterns <= 2:
            rows, cols = 1, 2
        elif n_patterns <= 4:
            rows, cols = 2, 2
        elif n_patterns <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Depth-Based Structured Light Patterns', fontsize=14, fontweight='bold')

        if n_patterns == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (result, title) in enumerate(zip(display_results, display_titles)):
            axes[i].imshow(result)
            axes[i].set_title(title)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_patterns, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        comparison_path = output_dir / f'{timestamp}_comparison.png'
        plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison: {comparison_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if args.depth:
        print(f"Depth source: External file ({args.depth})")
    else:
        print(f"Depth source: Generated from config geometry")
    print(f"Depth map: {depth_map.shape}")
    print(f"Patterns generated: {len(results)}")
    print(f"Output directory: {output_dir}")
    print("\nKey advantage: Works with external depth cameras!")
    print("  ✓ RealSense, Kinect, or any depth camera")
    print("  ✓ No 3D geometry knowledge needed")
    print("  ✓ Just load depth map and project patterns")
    print("=" * 70)


if __name__ == '__main__':
    main()
