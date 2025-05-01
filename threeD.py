import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

from config import get_cfg
from models import Generator, WireframeDiscriminator, renderer_g2v
from dataset import wireframeDataset_Rplan, generate_random_layout
from export_utils import export_layout_to_glb  # 3D export function

# ---- Config & Constants ----
room_types = {
    0: 'Living Room',
    1: 'Master Room',
    2: 'Kitchen',
    3: 'Bathroom',
    4: 'Second Room',
    5: 'Balcony'
}

room_colors = {
    0: 'skyblue',
    1: 'lightgreen',
    2: 'peachpuff',
    3: 'lightpink',
    4: 'khaki',
    5: 'lightgray'
}


# ---- Functions ----

def load_models(cfg, dataset, device, path):
    renderer = renderer_g2v(render_size=cfg.MODEL.RENDERER.RENDERING_SIZE, class_num=dataset.enc_len)
    generator = Generator(dataset=dataset).to(device)
    discriminator = WireframeDiscriminator(dataset=dataset, renderer=renderer, cfg=cfg).to(device)

    checkpoint_path = path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    print(f"✅ Loaded generator from: {checkpoint_path}")
    return generator


def generate_layout(generator, dataset, device):
    random_input_data = generate_random_layout(dataset, batch_size=1)
    room_type_input = torch.tensor(random_input_data[0]).to(device)
    room_count_input = torch.tensor([room_type_input.shape[1]]).to(device)

    with torch.no_grad():
        generated_output = generator(room_type_input, room_count_input)

    layout_tensor = generated_output[0][0].cpu().numpy()
    return layout_tensor


def visualize_layout(layout_tensor, epoch):
    room_labels = np.argmax(layout_tensor[:, :6], axis=1)
    df = pd.DataFrame(layout_tensor[:, -4:], columns=["x", "y", "w", "h"])
    df["room_type"] = [room_types.get(label, f"Room {label}") for label in room_labels]
    df["label_id"] = room_labels

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f"Generated Floor Plan - Epoch {epoch}")

    for idx, row in df.iterrows():
        color = room_colors.get(row["label_id"], "lightblue")
        rect = patches.Rectangle((row["x"], row["y"]), row["w"], row["h"],
                                 edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(row["x"] + row["w"]/2, row["y"] + row["h"]/2,
                row["room_type"], ha='center', va='center', fontsize=8)

    plt.show()


def export_layout_to_3d(layout_tensor, epoch, export_dir="glb_exports"):
    os.makedirs(export_dir, exist_ok=True)
    glb_path = os.path.join(export_dir, f"floorplan_epoch_{epoch:04d}.glb")
    export_layout_to_glb(layout_tensor, out_path=glb_path)


# ---- Main Execution ----

def main():
    epoch = 50
    enable_3d_export = True

    cfg = get_cfg()
    with open("real_dataset_full.pkl", "rb") as f:
        real_dataset = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    generator = load_models(cfg, real_dataset, device, epoch)
    layout_tensor = generate_layout(generator, real_dataset, device)

    visualize_layout(layout_tensor, epoch)

    if enable_3d_export:
        export_layout_to_3d(layout_tensor, epoch)


if __name__ == "__main__":
    main()
