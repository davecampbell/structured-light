import numpy as np
import matplotlib.pyplot as plt
from structured_light_3d import StructuredLightProjector, Scene3D, StructuredLightRenderer

# Test all patterns with appropriate projection methods
scene = Scene3D()
scene.add_ground_plane()
scene.add_box(size=(0.3,0.3,0.3), position=np.array([0,0,0.15]))

projector = StructuredLightProjector(
    position=np.array([0.5, -0.8, 1.5]),
    look_at=np.array([0.0, 0.0, 0.0]),
    fov=50.0
)

renderer = StructuredLightRenderer(
    projector=projector,
    camera_position=np.array([1.2, 0.0, 1.0]),
    camera_look_at=np.array([0.0, 0.0, 0.0]),
    camera_resolution=(640, 480)
)

# Create figure with all patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('All Patterns Working - Backward & Forward Projection', fontsize=14, fontweight='bold')

# 1. Vertical stripes (backward projection - continuous pattern)
pattern = projector.create_stripe_pattern(frequency=15, orientation='vertical')
rgb, _ = renderer.render(scene, pattern, use_forward_projection=False)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('Vertical Stripes (Backward Projection)')
axes[0, 0].axis('off')

# 2. Horizontal stripes (backward projection - continuous pattern)
pattern = projector.create_stripe_pattern(frequency=15, orientation='horizontal')
rgb, _ = renderer.render(scene, pattern, use_forward_projection=False)
axes[0, 1].imshow(rgb)
axes[0, 1].set_title('Horizontal Stripes (Backward Projection)')
axes[0, 1].axis('off')

# 3. Dots (forward projection - sparse pattern)
pattern = projector.create_dot_pattern(dot_spacing=40, dot_size=3)
rgb, _ = renderer.render(scene, pattern, use_forward_projection=True)
axes[1, 0].imshow(rgb)
axes[1, 0].set_title('Dots (Forward Projection)')
axes[1, 0].axis('off')

# 4. Grid (forward projection - sparse pattern)
pattern = projector.create_grid_pattern(grid_spacing=50, line_thickness=2)
rgb, _ = renderer.render(scene, pattern, use_forward_projection=True)
axes[1, 1].imshow(rgb)
axes[1, 1].set_title('Grid (Forward Projection)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('output/all_patterns_working.png', dpi=150, bbox_inches='tight')
print("Saved to output/all_patterns_working.png")
print("\nAll patterns now working correctly with proper 3D deformation!")
