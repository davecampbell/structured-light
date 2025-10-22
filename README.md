# Structured Light Generator

A Python tool for generating synthetic structured light patterns overlaid on images with depth data. This is designed for data augmentation to improve vision models for robotic grasping tasks.

## Overview

Structured light is a 3D scanning technique that projects known patterns onto objects and analyzes the deformation to extract depth information. This tool simulates various structured light patterns (like those from Kinect, laser scanners, or projector-camera systems) to create synthetic training data for computer vision models.

## Features

### Pattern Types

- **Sinusoidal Stripe Patterns**: Smooth vertical or horizontal fringes that deform based on depth
- **Binary Stripe Patterns**: Sharp black and white stripes with depth-based phase shifts
- **IR Dot Patterns**: Kinect-style infrared dot grids with depth displacement
- **Deformed Grid Patterns**: Regular grids warped by surface depth

### Capabilities

- **Depth-based Modulation**: All patterns react to depth maps, simulating real structured light behavior
- **Realistic Noise**: Adds Gaussian noise and missing data regions to simulate real sensor artifacts
- **RGB Overlay**: Blend patterns onto existing RGB images with configurable intensity
- **Cross-platform Display**: Automatically detects display availability
  - Shows interactive matplotlib window when display is available
  - Saves to timestamped PNG files when running headless (Linux servers, WSL, SSH sessions)
- **Output Management**: All visualizations saved to `output/` folder with timestamps

## Requirements

```bash
pip install numpy opencv-python matplotlib pillow
```

## Usage

### Basic Usage

```bash
python structured_light_generator.py
```

This will:
1. Generate a synthetic depth map with random box-like structures
2. Create multiple structured light patterns
3. Display a 3×3 grid of visualizations (if display available)
4. Save the visualization to `output/structured_light_patterns_YYYYMMDD_HHMMSS.png`

### Programmatic Usage

```python
from structured_light_generator import StructuredLightGenerator
import numpy as np

# Initialize generator
generator = StructuredLightGenerator(image_size=(640, 480))

# Load your depth map (or create synthetic one)
depth_map = np.load('your_depth_map.npy')

# Generate patterns
vertical_stripes = generator.create_stripe_pattern(
    depth_map,
    frequency=20.0,
    orientation='vertical'
)

dot_pattern = generator.create_dot_pattern(
    depth_map,
    dot_spacing=30,
    dot_size=3
)

# Add realistic noise
noisy_pattern = generator.add_realistic_noise(
    vertical_stripes,
    noise_level=0.05,
    missing_regions=True
)

# Overlay on RGB image
rgb_image = cv2.imread('your_image.jpg')
result = generator.overlay_on_rgb(rgb_image, vertical_stripes, blend_factor=0.3)
```

## Output

The script generates a 3×3 visualization grid showing:

### Row 1: Stripe Patterns
- Vertical sinusoidal stripes
- Horizontal sinusoidal stripes
- Binary stripes

### Row 2: Special Patterns
- IR dot pattern (Kinect-style)
- Deformed grid pattern
- Noisy pattern with missing regions

### Row 3: Context and Overlays
- Input depth map
- RGB + stripe pattern overlay
- RGB + dot pattern overlay

## Display Behavior

The script automatically detects whether a display is available:

- **With Display** (Mac, Linux desktop with X11): Opens interactive matplotlib window AND saves file
- **Without Display** (headless servers, SSH, WSL): Saves timestamped PNG to `output/` folder
- All output files are saved with timestamps like: `structured_light_patterns_20251022_083755.png`

## Use Cases

- **Data Augmentation**: Expand training datasets for depth-sensing neural networks
- **Sim-to-Real Transfer**: Train models on synthetic structured light before deploying to real sensors
- **Algorithm Testing**: Test structured light processing algorithms without hardware
- **Educational**: Understand how different structured light techniques work

## Customization

Adjust pattern parameters to match your specific needs:

```python
# Denser stripes
pattern = generator.create_stripe_pattern(depth_map, frequency=40.0)

# Larger dots
dots = generator.create_dot_pattern(depth_map, dot_spacing=50, dot_size=5)

# More realistic noise
noisy = generator.add_realistic_noise(pattern, noise_level=0.1, missing_regions=True)

# Stronger overlay
blended = generator.overlay_on_rgb(rgb, pattern, blend_factor=0.5)
```

## Next Steps

1. Load your actual depth maps and RGB images
2. Experiment with different pattern parameters
3. Save patterns as additional training channels
4. Integrate into your data pipeline

## License

MIT License - feel free to use in your projects!
