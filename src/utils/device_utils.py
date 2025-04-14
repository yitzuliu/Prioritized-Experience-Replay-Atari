"""
Device and system utilities for DQN with PER.

This module provides functions for detecting available computation devices
(CPU, CUDA, MPS) and retrieving system information.

DQN 優先經驗回放的設備和系統工具。

此模組提供用於檢測可用計算設備（CPU、CUDA、MPS）和獲取系統信息的功能。
"""

import platform
import os
import torch
import psutil
import numpy as np

# Add a variable to store the device
_device = None

def get_device():
    """
    Detect and return the best available device for PyTorch.
    
    This function checks for CUDA (NVIDIA) GPUs, Metal (Apple Silicon),
    and falls back to CPU if neither is available.
    
    Returns:
        torch.device: Best available PyTorch device
        
    檢測並返回 PyTorch 可用的最佳設備。
    
    此函數檢查 CUDA（NVIDIA）GPU、Metal（Apple Silicon），
    如果兩者都不可用，則退回到 CPU。
    
    返回：
        torch.device：最佳可用的 PyTorch 設備
    """
    global _device
    
    if _device is not None:
        return _device
    
    _device = torch.device("cpu")  # Default to CPU
    device_name = "CPU"
    
    # Always auto-detect the best available device
    # Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        
        # Configure for better performance on GPU
        torch.backends.cudnn.benchmark = True
        
    # Check for Apple Silicon (MPS - Metal Performance Shaders)
    elif (hasattr(torch.backends, "mps") and 
          torch.backends.mps.is_available() and 
          torch.backends.mps.is_built()):
        _device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"

    print(f"Using device: {device_name}")
    
    return _device

def get_gpu_memory_gb():
    """
    Get the total memory of the GPU in gigabytes.
    
    Returns:
        float: Total GPU memory in GB, or None if no GPU available
        
    獲取 GPU 總內存（以 GB 為單位）。
    
    返回：
        float：GPU 總內存（GB），如果沒有 GPU 則為 None
    """
    if torch.cuda.is_available():
        try:
            # Get memory in bytes, convert to GB
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            return None
    return None

def get_system_info():
    """
    Get detailed system information including hardware and software details.
    
    Returns:
        dict: System information including OS, CPU, GPU, and PyTorch version
        
    獲取詳細的系統信息，包括硬體和軟體詳情。
    
    返回：
        dict：系統信息，包括操作系統、CPU、GPU 和 PyTorch 版本
    """
    info = {
        'os': platform.system(),
        'os_version': platform.version(),
        'cpu_type': platform.processor() or "Unknown",
        'cpu_count': os.cpu_count(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }
    
    # Get memory information
    try:
        info['system_memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
    except:
        info['system_memory_gb'] = "Unknown"
    
    # Get GPU information if available
    if info['cuda_available']:
        try:
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = round(get_gpu_memory_gb(), 2)
        except:
            info['gpu_name'] = "Unknown CUDA device"
            info['gpu_memory_gb'] = "Unknown"
    
    return info