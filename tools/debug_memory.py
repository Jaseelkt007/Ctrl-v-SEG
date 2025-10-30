#!/usr/bin/env python3
"""
Memory debugging utility for tracking CUDA OOM issues.
Usage: Add this to your training script to monitor memory usage.
"""

import torch
import gc
from collections import defaultdict

class GPUMemoryMonitor:
    """Monitor and log GPU memory usage"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.peak_memory = 0
        self.checkpoints = []
        
    def log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def print_memory(self, tag=""):
        """Print detailed memory statistics"""
        if not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        if allocated > self.peak_memory:
            self.peak_memory = allocated
        
        self.log(f"[{tag}] GPU Memory:")
        self.log(f"  Allocated: {allocated:.3f} GB")
        self.log(f"  Reserved:  {reserved:.3f} GB")
        self.log(f"  Max Alloc: {max_allocated:.3f} GB")
        self.log(f"  Peak:      {self.peak_memory:.3f} GB")
        
        self.checkpoints.append({
            'tag': tag,
            'allocated': allocated,
            'reserved': reserved
        })
    
    def print_tensor_memory(self):
        """Print memory usage by tensor type"""
        tensor_sizes = defaultdict(lambda: {'count': 0, 'memory': 0})
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        size = obj.element_size() * obj.nelement() / 1024**3
                        dtype = str(obj.dtype)
                        shape = str(tuple(obj.shape))
                        key = f"{dtype}_{shape}"
                        tensor_sizes[key]['count'] += 1
                        tensor_sizes[key]['memory'] += size
            except:
                pass
        
        self.log("\nTensor Memory Usage (CUDA tensors only):")
        sorted_tensors = sorted(tensor_sizes.items(), 
                              key=lambda x: x[1]['memory'], 
                              reverse=True)[:20]
        
        for key, stats in sorted_tensors:
            self.log(f"  {key}: {stats['count']} tensors, {stats['memory']:.3f} GB")
    
    def reset_peak_memory(self):
        """Reset peak memory tracking"""
        torch.cuda.reset_peak_memory_stats()
        self.peak_memory = torch.cuda.memory_allocated() / 1024**3
    
    def get_summary(self):
        """Get summary of all checkpoints"""
        self.log("\n=== Memory Usage Summary ===")
        for cp in self.checkpoints:
            self.log(f"{cp['tag']}: Allocated={cp['allocated']:.3f}GB, Reserved={cp['reserved']:.3f}GB")


def optimize_memory():
    """Call this to aggressively free memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def check_memory_available(required_gb=1.0):
    """Check if there's enough memory available"""
    if not torch.cuda.is_available():
        return False
    
    free_memory = (torch.cuda.get_device_properties(0).total_memory - 
                   torch.cuda.memory_allocated()) / 1024**3
    return free_memory >= required_gb


def print_model_memory(model, model_name="Model"):
    """Print memory usage of model parameters"""
    total_params = 0
    trainable_params = 0
    param_memory = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        param_memory += param.element_size() * num_params
    
    print(f"\n{model_name} Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter memory: {param_memory / 1024**3:.3f} GB")


if __name__ == "__main__":
    # Example usage
    monitor = GPUMemoryMonitor()
    monitor.print_memory("Initial")
    
    # Create some tensors
    x = torch.randn(1000, 1000, device='cuda')
    monitor.print_memory("After creating tensor")
    
    del x
    optimize_memory()
    monitor.print_memory("After cleanup")
    
    monitor.print_tensor_memory()
    monitor.get_summary()
