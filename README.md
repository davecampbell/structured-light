# Structured Light Generator

A Python tool for generating synthetic structured light patterns overlaid on images with depth data. This is designed for data augmentation to improve vision models for robotic grasping tasks.

## Two Approaches Available

### 1. Simple 2D Pattern Generator (`structured_light_generator.py`)
Quick depth-based pattern overlay for existing images.

![Structured Light Patterns](output/structured_light_patterns_20251022_083755.png)

### 2. Full 3D Simulator (`structured_light_3d.py`) **NEW!**
Realistic 3D scene rendering with projector and camera at different positions.

![3D Structured Light](output/structured_light_3d_vertical_stripes_20251023_115204.png)

## Overview

Structured light is a 3D scanning technique that projects known patterns onto objects and analyzes the deformation to extract depth information. This tool provides two ways to generate synthetic structured light training data:

1. **2D Pattern Generator**: Fast pattern overlay onto existing depth maps
2. **3D Simulator**: Full geometric simulation with configurable projector and camera positions

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
pip install -r requirements.txt
```

Or manually:
```bash
# For 2D pattern generator
pip install numpy opencv-python matplotlib pillow

# For 3D simulator (includes above)
pip install numpy opencv-python matplotlib pillow trimesh pyrender
```

## Usage

### Quick Start: 2D Pattern Generator

```bash
python structured_light_generator.py
```

This will:
1. Generate a synthetic depth map with random box-like structures
2. Create multiple structured light patterns
3. Display a 3×3 grid of visualizations (if display available)
4. Save the visualization to `output/structured_light_patterns_YYYYMMDD_HHMMSS.png`

### Quick Start: 3D Simulator

```bash
python structured_light_3d.py
```

This will:
1. Create a 3D scene with random boxes on a ground plane
2. Position projector at (0.5, -0.8, 1.5) and camera at (1.2, 0.0, 1.0)
3. Render the scene with multiple structured light patterns
4. Generate both RGB and depth images from camera viewpoint
5. Save visualizations to `output/structured_light_3d_*.png`

### Programmatic Usage: 2D Pattern Generator

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

### Programmatic Usage: 3D Simulator

```python
from structured_light_3d import (
    StructuredLightProjector,
    Scene3D,
    StructuredLightRenderer
)
import numpy as np

# Create projector at custom position
projector = StructuredLightProjector(
    position=np.array([0.5, -0.8, 1.5]),
    look_at=np.array([0.0, 0.0, 0.0]),
    fov=50.0,
    resolution=(1024, 768)
)

# Build 3D scene
scene = Scene3D()
scene.add_ground_plane(size=3.0, height=-0.5)
scene.add_box(
    size=(0.2, 0.3, 0.15),
    position=np.array([0.0, 0.0, 0.1])
)
# Or generate random boxes
scene.generate_random_boxes(num_boxes=5, bounds=(1.0, 1.0, 0.4))

# Setup renderer with camera
renderer = StructuredLightRenderer(
    projector=projector,
    camera_position=np.array([1.2, 0.0, 1.0]),
    camera_look_at=np.array([0.0, 0.0, 0.0]),
    camera_resolution=(640, 480),
    camera_fov=60.0
)

# Generate pattern and render
pattern = projector.create_stripe_pattern(frequency=15, orientation='vertical')
rgb_image, depth_map = renderer.render(
    scene,
    pattern,
    ambient_light=0.3,
    pattern_intensity=0.6
)

# rgb_image and depth_map are now ready to use for training
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
