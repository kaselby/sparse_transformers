"""
Device utilities for cross-platform support (CUDA, MPS, CPU).

This module provides unified device handling for NVIDIA GPUs (CUDA),
Apple Silicon GPUs (MPS), and CPU fallback.
"""

import torch
from typing import Optional


def get_device(device_arg: str = "auto") -> torch.device:
    """
    Get the appropriate device based on availability and user preference.

    Args:
        device_arg: Device specification. Can be:
            - "auto": Automatically select best available (mps > cuda > cpu)
            - "cuda": Force CUDA (falls back to CPU if unavailable)
            - "mps": Force MPS (falls back to CPU if unavailable)
            - "cpu": Force CPU
            - Specific device like "cuda:0" or "cuda:1"

    Returns:
        torch.device: The selected device

    Examples:
        >>> device = get_device("auto")  # Auto-select best device
        >>> device = get_device("cuda")  # Prefer CUDA
        >>> device = get_device("mps")   # Prefer MPS (Apple Silicon)
    """
    if device_arg == "auto":
        # Priority: MPS > CUDA > CPU
        # MPS is preferred on Apple Silicon for local development
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    elif device_arg == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        else:
            print("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    else:
        # Allow specific device specifications like "cuda:0", "cpu", etc.
        return torch.device(device_arg)


def empty_cache(device: torch.device) -> None:
    """
    Empty the memory cache for the given device.

    Args:
        device: The device whose cache should be emptied

    Examples:
        >>> device = torch.device("mps")
        >>> empty_cache(device)
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    # CPU doesn't have a cache to empty


def get_device_name(device: torch.device) -> str:
    """
    Get a human-readable name for the device.

    Args:
        device: The device to name

    Returns:
        str: Human-readable device name

    Examples:
        >>> device = torch.device("mps")
        >>> print(get_device_name(device))
        Apple Silicon GPU (MPS)
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            return f"NVIDIA {torch.cuda.get_device_name(device)}"
        return "CUDA (unavailable)"
    elif device.type == "mps":
        return "Apple Silicon GPU (MPS)"
    elif device.type == "cpu":
        return "CPU"
    else:
        return str(device)


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.

    Returns:
        bool: True if CUDA or MPS is available
    """
    return torch.cuda.is_available() or (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    )


def get_available_devices() -> list[str]:
    """
    Get a list of all available device types.

    Returns:
        list[str]: List of available device types (e.g., ['cuda', 'cpu'])
    """
    devices = ['cpu']

    if torch.cuda.is_available():
        devices.insert(0, 'cuda')

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices.insert(0, 'mps')

    return devices


def synchronize(device: torch.device) -> None:
    """
    Synchronize the device (wait for all operations to complete).

    Args:
        device: The device to synchronize

    Note:
        MPS doesn't have an explicit synchronize function, so we use
        a CPU synchronization point instead.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # MPS doesn't have explicit synchronization
        # Creating a CPU tensor from MPS tensor forces synchronization
        torch.mps.synchronize()


def supports_autocast(device: torch.device) -> bool:
    """
    Check if the device supports automatic mixed precision (autocast).

    Args:
        device: The device to check

    Returns:
        bool: True if autocast is supported
    """
    if device.type == "cuda":
        return True
    elif device.type == "mps":
        # MPS has limited autocast support
        return False
    else:
        return False
