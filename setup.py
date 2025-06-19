#!/usr/bin/env python3
"""
简化的 setup.py，解决 PyTorch 兼容性问题
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

def get_cuda_version():
    """获取 CUDA 版本"""
    try:
        return torch.version.cuda
    except:
        return "unknown"

def get_compute_capability():
    """获取 GPU 计算能力"""
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return f"sm_{cap[0]}{cap[1]}"
        else:
            return "sm_70"  # 默认
    except:
        return "sm_70"

def main():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {get_cuda_version()}")
    print(f"GPU 计算能力: {get_compute_capability()}")
    
    # 基本编译参数
    extra_compile_args = {
        'cxx': [
            '-O3',
            '-std=c++17',
            '-DWITH_CUDA',
        ],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            f'-arch={get_compute_capability()}',
            '-DWITH_CUDA',
            '--expt-relaxed-constexpr',  # 允许 constexpr 的放松约束
        ]
    }
    
    # 兼容旧版本 PyTorch
    torch_version = torch.__version__.split('.')
    major, minor = int(torch_version[0]), int(torch_version[1])
    
    if major >= 2 or (major == 1 and minor >= 12):
        # 新版本 PyTorch
        extra_compile_args['nvcc'].append('-DTORCH_VERSION_MAJOR=2')
    else:
        # 旧版本 PyTorch
        extra_compile_args['nvcc'].append('-DTORCH_VERSION_MAJOR=1')
        extra_compile_args['cxx'].append('-DTORCH_VERSION_MAJOR=1')
    
    setup(
        name='cuda_vel_forward_c',
        ext_modules=[
            CUDAExtension(
                name='cuda_vel_forward_c',
                sources=[
                    'cuda_forward.cpp',
                    'cuda_kernels.cu',
                ],
                extra_compile_args=extra_compile_args,
                include_dirs=[],
                libraries=['cudart', 'cublas'],
                library_dirs=[],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        },
        zip_safe=False,
    )

if __name__ == '__main__':
    main() 