import sys
import numpy as np
from tqdm import tqdm
import torch
import re
from pathlib import Path
from datetime import datetime
from network import ResNetUNet
from predict import *
import matplotlib.pyplot as plt

sys.path.append('..')
from notebooks.video_io import load_frames, write_sequence_to_mp4, show_breakers_on_frame

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Initialize model
model = ResNetUNet(n_class=1)

# Load the saved state dictionary
# state_dict = torch.load('best_model_20241105.pt', map_location=device)
# state_dict = torch.load('best_model_20241119.pt', map_location=device)
# state_dict = torch.load('best_model_20241120.pt', map_location=device)
# state_dict = torch.load('best_model_20241128.pt', map_location=device)
# state_dict = torch.load('best_model_20241202.pt', map_location=device)
state_dict = torch.load('best_model_20241203.pt', map_location=device)

# Load state dictionary into model
model.load_state_dict(state_dict)

# Move model to device
model = model.to(device)

# Set model to evaluation mode if you're going to use it for inference
model.eval()

# List of video paths and their slice ranges
video_configs = [
    ('/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211014T123100Z_RectifiedVideo.avi', (0, 1000)),
    ('/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211014T120100Z_RectifiedVideo.avi', (1500, 2500)),
    ('/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211013T170100Z_RectifiedVideo.avi', (0, 250)),
    ('/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211028T153100Z_RectifiedVideo.avi', (2000, 3000)),
    ('/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211014T130100Z_RectifiedVideo.avi', (1000, 2000))
]

def get_datetime_from_path(path):
    pattern = r'\d{8}T\d{6}Z'
    datetime_str = re.search(pattern, path).group()
    return datetime_str

# Set up device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Create results directory if it doesn't exist
results_dir = Path('results/20241203')
results_dir.mkdir(parents=True, exist_ok=True)

# Process each video
for video_path, slice_range in video_configs:
    print(f"\nProcessing {video_path}")
    
    # Load frames with optional slicing
    ts = load_frames(video_path)
    if slice_range is not None:
        ts = ts[slice_range[0]:slice_range[1]]
    
    # Initialize breakers array
    breakers = np.zeros_like(ts, dtype=float)
    
    # Process each frame
    for i in tqdm(range(ts.shape[1])):  # Assuming ts shape is (width, frames, height)
        img = ts[:, i, :].T/255.0
        b = predict_large_image_patchify(model, img, patch_size=128, overlap=16, batch_size=64, device=device).T
        breakers[:, i, :] = b
    
    # Get datetime string from filename
    datetime_str = get_datetime_from_path(video_path)
    
    # Save results
    write_sequence_to_mp4((breakers > 0.5).astype(np.uint8) * 255, 
                         str(results_dir / f'{datetime_str}_breakers.avi'), 
                         FPS=2, 
                         color=False,
                         lossless=True)
    
    write_sequence_to_mp4((breakers > 0.5).astype(np.uint8) * 255, 
                         str(results_dir / f'{datetime_str}_breakers.mp4'), 
                         FPS=2, 
                         color=False)
    
    write_sequence_to_mp4((ts).astype(np.uint8), 
                         str(results_dir / f'{datetime_str}_breaker_highlight.mp4'), 
                         fn=show_breakers_on_frame, 
                         FPS=2, 
                         data=breakers > 0.5)
    
    
    fig, ax = plt.subplots(1,1, figsize=(24, 8))
    count = (breakers>0.5).sum(0)
    im = ax.imshow(count.T / len(breakers) * 2, cmap='viridis')
    plt.colorbar(im, ax=ax, label='breakers per seconds')
    ax.set_title(get_datetime_from_path(video_path))
    plt.savefig(f'{str(results_dir)}/{get_datetime_from_path(video_path)}.png')
    plt.clf()
    plt.close()

    
    
    print(f"Completed processing {datetime_str}")