import sys
import numpy as np
from tqdm import tqdm
import torch
import re
import xarray as xr
from pathlib import Path
from datetime import datetime
from network import ResNetUNet
from predict import *
import matplotlib.pyplot as plt

sys.path.append('..')
from notebooks.video_io import load_frames, write_sequence_to_mp4, show_breakers_on_frame

def find_best_segment(video_data, min_segment_length=250, threshold=0.2):
    """
    Find the longest continuous segment without significant missing data.
    
    Args:
        video_data: numpy array of video frames
        min_segment_length: minimum length of segment to consider
        threshold: maximum fraction of zeros allowed in a frame to consider it valid
        
    Returns:
        tuple: (start_idx, end_idx) of best segment
    """
    # Calculate fraction of zeros in each frame
    zero_fractions = (video_data == 0).mean(axis=(1, 2))
    
    # Mark frames with too many zeros as invalid
    valid_frames = zero_fractions < threshold
    
    # Find continuous segments
    changes = np.diff(valid_frames.astype(int))
    segment_starts = np.where(changes == 1)[0] + 1
    segment_ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if valid_frames[0]:
        segment_starts = np.insert(segment_starts, 0, 0)
    if valid_frames[-1]:
        segment_ends = np.append(segment_ends, len(valid_frames))
    
    # Calculate segment lengths
    segment_lengths = segment_ends - segment_starts
    
    # Find longest segment meeting minimum length
    valid_segments = segment_lengths >= min_segment_length
    if not np.any(valid_segments):
        return None, None
    
    best_idx = np.argmax(segment_lengths * valid_segments)
    return segment_starts[best_idx], segment_ends[best_idx]

# Set device and initialize model
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = ResNetUNet(n_class=1)
state_dict = torch.load('best_model_20241203.pt', map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Load wave data
wave_data = xr.load_dataset('../../data/FRF-ocean_waves_8m-array_202110.nc')

def get_wave_conditions(timestamp):
    """Get wave conditions for a specific timestamp."""
    wave_data_point = wave_data.sel(time=timestamp, method='nearest')
    return {
        'Hs': float(wave_data_point.waveHs),
        'direction': float(wave_data_point.waveMeanDirection),
        'spread': float(wave_data_point.directionalPeakSpread)
    }

def get_datetime_from_path(path):
    """Extract datetime from filepath and convert to datetime object."""
    pattern = r'(\d{8}T\d{6}Z)'
    datetime_str = re.search(pattern, path).group()
    return datetime.strptime(datetime_str, '%Y%m%dT%H%M%SZ')

# Video configurations
video_configs = [
    # '/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211010T150100Z_RectifiedVideo.avi',
    '/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211011T140100Z_RectifiedVideo.avi',
    '/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211012T133100Z_RectifiedVideo.avi',
    '/Users/mleclair/phd/code/dunex/data/argus/ArgusFF_20211026T160100Z_RectifiedVideo.avi'
]

# Create results directory
results_dir = Path('results/20241203/test/full')
results_dir.mkdir(parents=True, exist_ok=True)


find_best = False

# Process each video
for video_path in video_configs:
    print(f"\nProcessing {video_path}")
    
    # Get timestamp and wave conditions
    timestamp = get_datetime_from_path(video_path)
    wave_conditions = get_wave_conditions(timestamp)
    
    # Load full video first
    print("Loading video...")
    ts = load_frames(video_path)
    
    # Find best segment
    start_idx = 0
    end_idx = len(ts)-1
    if find_best:
        print("Finding best segment...")
        start_idx, end_idx = find_best_segment(ts)
    
    if start_idx is None:
        print(f"No suitable segment found in {video_path}")
        continue
        
    print(f"Best segment found: frames {start_idx} to {end_idx}")
    
    # Slice to best segment
    ts = ts[start_idx:end_idx,:, :]
    
    # Initialize breakers array
    breakers = np.zeros_like(ts, dtype=float)
    
    # Process each frame
    print("Processing frames...")
    for i in tqdm(range(ts.shape[1])):
        img = ts[:, i, :].T/255.0
        b = predict_large_image_patchify(model, img, patch_size=128, overlap=16, batch_size=64, device=device).T
        breakers[:, i, :] = b
    
    # Generate output filenames
    datetime_str = timestamp.strftime('%Y%m%dT%H%M%SZ')
    
    # Save video outputs
    print("Saving outputs...")
    write_sequence_to_mp4((breakers > 0.5).astype(np.uint8) * 255, 
                         str(results_dir / f'{datetime_str}_breakers.avi'), 
                         FPS=2, 
                         color=False,
                         lossless=True)
    
    write_sequence_to_mp4((ts).astype(np.uint8), 
                         str(results_dir / f'{datetime_str}_breaker_highlight.mp4'), 
                         fn=show_breakers_on_frame, 
                         FPS=2, 
                         data=breakers > 0.5)
    
    # Create breaker density plot with wave conditions
    fig, ax = plt.subplots(1, 1, figsize=(24, 8))
    count = (breakers > 0.5).sum(0).T  # Sum along time axis
    
    # Plot breaker density
    im = ax.imshow(count / len(ts) * 2, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Breakers per second')
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 500)
    # Add wave conditions and frame range to title
    title = (f"Wave Conditions at {timestamp.strftime('%Y-%m-%d %H:%M UTC')} (Frames {start_idx}-{end_idx}) [{(end_idx - start_idx) / 2 / 60:.2f}min]\n"
            f"Hs: {wave_conditions['Hs']:.2f}m | "
            f"Direction: {wave_conditions['direction']:.1f}° | "
            f"Spread: {wave_conditions['spread']:.1f}°")
    ax.set_title(title)
    
    # Save figure
    plt.savefig(str(results_dir / f'{datetime_str}_density.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()
    
    print(f"Completed processing {datetime_str}")