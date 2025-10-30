#!/usr/bin/env python3
"""
Get Camera Intrinsics from Intel RealSense

This script shows how to extract camera intrinsics from a RealSense depth camera
and format them for use in the config file.

Requirements:
    pip install pyrealsense2

Usage:
    python get_realsense_intrinsics.py
"""

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed")
    print("Install with: pip install pyrealsense2")
    exit(1)

import numpy as np


def get_realsense_intrinsics():
    """Get intrinsics from connected RealSense camera."""

    print("=" * 70)
    print("RealSense Camera Intrinsics Extraction")
    print("=" * 70)
    print()

    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error: Could not start RealSense pipeline")
        print(f"Make sure a RealSense camera is connected")
        print(f"Details: {e}")
        return

    try:
        # Get depth stream profile
        depth_profile = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        print("✓ Successfully connected to RealSense camera")
        print()
        print("=" * 70)
        print("Camera Intrinsics")
        print("=" * 70)
        print(f"Resolution: {depth_intrinsics.width} x {depth_intrinsics.height}")
        print(f"Focal Length X (fx): {depth_intrinsics.fx:.2f} pixels")
        print(f"Focal Length Y (fy): {depth_intrinsics.fy:.2f} pixels")
        print(f"Principal Point X (cx): {depth_intrinsics.ppx:.2f} pixels")
        print(f"Principal Point Y (cy): {depth_intrinsics.ppy:.2f} pixels")
        print(f"Distortion Model: {depth_intrinsics.model}")
        print(f"Distortion Coeffs: {depth_intrinsics.coeffs}")
        print()

        print("=" * 70)
        print("YAML Config Format")
        print("=" * 70)
        print()
        print("camera:")
        print("  position: [1.2, 0.0, 1.0]  # Adjust to your setup")
        print("  look_at: [0.0, 0.0, 0.0]")
        print(f"  resolution: [{depth_intrinsics.width}, {depth_intrinsics.height}]")
        print()
        print("  # Camera intrinsics from RealSense")
        print("  intrinsics:")
        print(f"    fx: {depth_intrinsics.fx:.2f}")
        print(f"    fy: {depth_intrinsics.fy:.2f}")
        print(f"    cx: {depth_intrinsics.ppx:.2f}")
        print(f"    cy: {depth_intrinsics.ppy:.2f}")
        print()

        print("=" * 70)
        print("Copy the above YAML block into your config file!")
        print("=" * 70)
        print()

        # Get a sample depth frame
        print("Capturing sample depth frame...")
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if depth_frame:
            depth_map = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_map = depth_map / 1000.0  # Convert mm to meters

            valid_depth = depth_map[depth_map > 0]
            if len(valid_depth) > 0:
                print(f"✓ Sample depth captured")
                print(f"  Depth range: {valid_depth.min():.3f} - {valid_depth.max():.3f} meters")
                print(f"  Valid pixels: {len(valid_depth)} / {depth_map.size}")
                print()
                print("To save this depth map:")
                print()
                print("  from structured_light_from_depth import DepthMapIO")
                print("  DepthMapIO.save_depth_npy(depth_map, 'realsense_depth.npy')")
                print()

    finally:
        pipeline.stop()
        print("Camera stopped")


if __name__ == '__main__':
    get_realsense_intrinsics()
