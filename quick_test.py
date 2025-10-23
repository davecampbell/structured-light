import numpy as np
from structured_light_3d import StructuredLightProjector, Scene3D, StructuredLightRenderer

# Simple test
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

pattern = projector.create_stripe_pattern(frequency=15, orientation='vertical')
print(f"Pattern shape: {pattern.shape}, dtype: {pattern.dtype}, min: {pattern.min()}, max: {pattern.max()}")

rgb, depth = renderer.render(scene, pattern, use_forward_projection=False)
print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}, min: {rgb.min()}, max: {rgb.max()}")

# Save
import matplotlib.pyplot as plt
plt.imsave('output/quick_test.png', rgb)
print("Saved to output/quick_test.png")
