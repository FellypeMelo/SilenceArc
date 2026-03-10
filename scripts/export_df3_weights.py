import os
import torch
import json
import numpy as np

def export_weights(checkpoint_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # In DeepFilterNet checkpoints, state_dict is often nested
    state_dict = checkpoint.get('state_dict', checkpoint)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    
    metadata = {}
    
    for name, param in state_dict.items():
        # Sanitize name for filename
        filename = name.replace('.', '_') + ".bin"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to numpy and save as binary
        data = param.detach().cpu().numpy().astype(np.float32)
        data.tofile(filepath)
        
        metadata[name] = {
            "file": filename,
            "shape": list(data.shape),
            "dtype": "float32"
        }
        print(f"Exported {name} (shape: {data.shape}) to {filename}")

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Exported {len(state_dict)} tensors to {output_dir}")

if __name__ == "__main__":
    checkpoint_path = "DeepFilterNet/models/DeepFilterNet3/DeepFilterNet3/checkpoints/model_120.ckpt.best"
    output_dir = "models/df3_weights"
    export_weights(checkpoint_path, output_dir)
