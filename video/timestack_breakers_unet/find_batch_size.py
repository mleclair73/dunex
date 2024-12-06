from torch.utils.data import DataLoader, random_split
import torch
import time
from tqdm import tqdm
import numpy as np
from loss import calc_loss
from combined_dataset import DunexDataset
from network import ResNetUNet
from tqdm import tqdm 
from collections import defaultdict
# Your existing setup

samples_per_image = 64
ds = DunexDataset(['cvat_annotations/triplets_fully_labeled/annotations_poly_updated.xml', 
                   'cvat_annotations/additional_data/annotations_poly.xml'],
                  samples_per_image = samples_per_image)
train, val, test = torch.utils.data.random_split(ds, (225 * samples_per_image, 25 * samples_per_image, 7 * samples_per_image)) # 257 (* 3) * 32

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

num_class = 1
model = ResNetUNet(num_class).to(device)

def benchmark_batch_size(dataset, model, device, batch_sizes, num_iterations=1):
    """
    Benchmark different batch sizes to find optimal training speed.
    
    Args:
        dataset: PyTorch dataset
        model: PyTorch model
        device: torch device (cpu or mps)
        batch_sizes: list of batch sizes to test
        num_iterations: number of training iterations for each batch size
    
    Returns:
        dict: Results containing times and memory usage for each batch size
    """
    results = {}
    model.train()
    
    for batch_size in tqdm(batch_sizes):
        print(batch_size)
        try:
            # Create dataloader with current batch size
            print('Creating Dataloader')
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # Important for MPS device
            )
            print('Done')
            # Initialize optimizer
            optimizer = torch.optim.Adam(model.parameters())
            
            # Warmup
            for batch in dataloader:
                break
                
            # Timing
            start_time = time.time()
            samples_processed = 0
            
            print('Iterations')
            metrics = defaultdict(float)
            for i in tqdm(range(num_iterations)):
                for inputs, targets in dataloader:
                    inputs = inputs.to(device)
                    labels = targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics, i, start_boundary_epoch=0)
                    # loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    samples_processed += inputs.size(0)
                    
                    if i >= num_iterations:
                        break
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            samples_per_second = samples_processed / total_time
            
            # Get memory usage
            if device.type == "mps":
                # MPS doesn't support memory stats directly, use system memory as proxy
                memory_used = torch.mps.current_allocated_memory() / 1024**2  # MB
            else:
                memory_used = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
            
            results[batch_size] = {
                'time': total_time,
                'samples_per_second': samples_per_second,
                'memory_used_mb': memory_used
            }
            
            # Clear memory
            torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"Error with batch size {batch_size}: {str(e)}")
            break
            
    return results

def find_optimal_batch_size(dataset, model, device):
    """
    Find and print the optimal batch size based on benchmarking results.
    """
    # Test various batch sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    results = benchmark_batch_size(dataset, model, device, batch_sizes)
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Batch Size':^10} | {'Time (s)':^12} | {'Samples/s':^12} | {'Memory (MB)':^12}")
    print("-" * 80)
    
    for batch_size, metrics in results.items():
        print(f"{batch_size:^10} | {metrics['time']:>12.2f} | {metrics['samples_per_second']:>12.2f} | {metrics['memory_used_mb']:>12.2f}")
    
    # Find optimal batch size (highest samples/second)
    optimal_batch = max(results.items(), key=lambda x: x[1]['samples_per_second'])[0]
    
    print("\nRecommendation:")
    print(f"Optimal batch size for your setup: {optimal_batch}")
    print(f"Samples per second: {results[optimal_batch]['samples_per_second']:.2f}")
    print(f"Memory usage: {results[optimal_batch]['memory_used_mb']:.2f} MB")
    
    return optimal_batch

if __name__ == "__main__":
    # Find optimal batch size
    optimal_batch_size = find_optimal_batch_size(train, model, device)