from flask import Flask, request, send_file
import torch
import pickle
import os
import tempfile

from threeD import load_models, generate_layout
from export_utils import export_layout_to_glb
from config import get_cfg
import uuid
import gdown

import threading
import time

EXPORT_DIR = "api_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)
EXPIRY_SECONDS = 60          # Delete files older than 60 seconds
CLEANUP_INTERVAL = 60 * 10     # Run cleanup every 60 seconds

def cleanup_old_exports():
    while True:
        now = time.time()
        deleted = 0
        for fname in os.listdir(EXPORT_DIR):
            fpath = os.path.join(EXPORT_DIR, fname)
            if os.path.isfile(fpath):
                age = now - os.path.getmtime(fpath)
                if age > EXPIRY_SECONDS:
                    try:
                        os.remove(fpath)
                        print(f"🧹 Deleted expired file: {fname}")
                        deleted += 1
                    except Exception as e:
                        print(f"⚠️ Failed to delete {fname}: {e}")
        print(f"🧼 Cleanup cycle complete. {deleted} file(s) deleted.")
        time.sleep(CLEANUP_INTERVAL)

# === Download model checkpoint from Google Drive if missing ===
def download_checkpoint_if_needed(file_path, gdrive_id):
    if not os.path.exists(file_path):
        print("⬇️ Downloading model checkpoint from Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        output = gdown.download(url, file_path, quiet=False)
        print(f"✅ Downloaded: {output}")
        return output
    else:
        print("✅ Checkpoint already present.")
        return file_path



app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config and dataset once on startup
cfg = get_cfg()
with open("real_dataset_full.pkl", "rb") as f:
    real_dataset = pickle.load(f)

print("✅ Dataset loaded.")
print("✅ Configuration loaded.")

# DOWNLOAD THE CHECKPOINT
gdrive_id = "13yZQiVQz04aoyH3NDH6WzxpUtba5R4dh" #EPOCH 650
checkpoint_path = f"params_rplan_epoch.pkl"
downloaded_file = download_checkpoint_if_needed(checkpoint_path, gdrive_id)
print("Final file path:", downloaded_file)

 # Load model and generate layout
generator = load_models(cfg, real_dataset, device, checkpoint_path)


@app.route('/generate', methods=['GET'])
def generate_floorplan_glb():
    try:
        export_dir = EXPORT_DIR
        # Unique filename
        uid = str(uuid.uuid4())[:8]
        glb_path = os.path.join(export_dir, f"floorplan_{uid}.glb")

       
        layout_tensor = generate_layout(generator, real_dataset, device)

        # Export to .glb only
        export_layout_to_glb(layout_tensor, out_path=glb_path)

       # Send the file
        return send_file(glb_path, as_attachment=True, mimetype='model/gltf-binary')

    except Exception as e:
        return {"status": "error", "message": str(e)}, 500
if __name__ == "__main__":
    threading.Thread(target=cleanup_old_exports, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
