import numpy as np
import os
import trimesh
from trimesh.scene import Scene
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors  # Correct import for color conversion

# Room type labels
room_types = {
    0: 'Living Room',
    1: 'Master Room',
    2: 'Kitchen',
    3: 'Bathroom',
    4: 'Second Room',
    5: 'Balcony'
}

# Corresponding room colors
room_colors = {
    0: 'skyblue',
    1: 'lightgreen',
    2: 'peachpuff',
    3: 'lightpink',
    4: 'khaki',
    5: 'lightgray'
}

def export_layout_to_glb(layout, out_path="floorplan.glb", wall_height=1):
    """
    Converts a floor plan layout tensor (N, 10) to a colored 3D .glb file.
    Each room is a box with height and color based on room type.
    """
    geometry = layout[:, -4:]
    room_labels = np.argmax(layout[:, :6], axis=1)

    scene = Scene()

    for i, (x, y, w, h) in enumerate(geometry):
        label = int(room_labels[i])
        room_name = room_types.get(label, f"Room{label}")

        center_x = x + w / 2
        center_y = y + h / 2

        # Create a 3D box
        box = trimesh.creation.box(extents=[w, wall_height, h])
        box.apply_translation([center_x, wall_height / 2, center_y])

        # Apply consistent color per room type
        color_name = room_colors.get(label, "lightblue")
        rgba = np.array(mcolors.to_rgba(color_name)) * 255
        rgba = rgba.astype(np.uint8)
        vertex_colors = np.tile(rgba, (box.vertices.shape[0], 1))
        box.visual.vertex_colors = vertex_colors

        # Add the box to the scene
        scene.add_geometry(box, node_name=room_name)

    # Center model around origin (fixes offset in Unity)
    scene.apply_translation(-scene.centroid)

    # Align bottom of model to Y = 0 (grounded on AR plane)
    min_y = scene.bounds[0][1]  # lowest Y value
    scene.apply_translation([0, -min_y, 0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scene.export(out_path)
    print(f"✅ Exported 3D model with room-type colors: {out_path}")

def export_layout_to_png(layout, out_path="floorplan.png"):
    """
    Converts a floor plan layout tensor (N, 10) to a 2D labeled .png image.
    Each room is shown as a rectangle with a color and name.
    """
    geometry = layout[:, -4:]
    room_labels = np.argmax(layout[:, :6], axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title("Generated Floor Plan")

    for i, (x, y, w, h) in enumerate(geometry):
        label = int(room_labels[i])
        room_name = room_types.get(label, f"Room{label}")
        color = room_colors.get(label, 'lightblue')

        rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                 edgecolor='black', facecolor=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, room_name, fontsize=7,
                ha='center', va='center')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"🖼️ Exported 2D image: {out_path}")
