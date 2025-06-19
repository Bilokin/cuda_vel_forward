"""
CUDA 加速的声波正演模拟包
"""

__version__ = "1.0.0"

from .cuda_vel_forward import (
    Vel_Forward
)

__all__ = [
    'Vel_Forward'
]

