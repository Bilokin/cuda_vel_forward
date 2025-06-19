"""
CUDA 加速的声波正演模拟 - Python 接口
提供与 py_vel_to_seis.py 中 vel_to_seis_batch 相同的接口
"""

import time
import torch
import numpy as np
from typing import Union, Optional

import torch.nn as nn

# 尝试导入编译的 CUDA 扩展
try:
    import cuda_vel_forward_c
    CUDA_AVAILABLE = True
    print("✅ CUDA 扩展已加载")
except ImportError:
    CUDA_AVAILABLE = False
    print("❌ CUDA 扩展未找到，尝试运行 build.py 进行编译")

DTYPE  = torch.float32

# ------------------------------------------------------------
# 1. 工具函数
# ------------------------------------------------------------
def ricker(f, dt, nt, *, device):
    """返回 (nt,) 的 Ricker 波子"""
    nw   = int(2.2 / f / dt)
    nw   = 2 * (nw // 2) + 1
    nc   = nw // 2 + 1
    k    = torch.arange(1, nw + 1, device=device, dtype=DTYPE)
    beta = ((nc - k) * f * dt * torch.pi) ** 2
    w0   = (1.0 - 2.0 * beta) * torch.exp(-beta)

    w    = torch.zeros(nt, device=device, dtype=DTYPE)
    w[: w0.numel()] = w0
    return w                                          # (nt,)



def padvel_bchw(v4d, nbc):
    """
    v : (H, W)  → replicate-pad → (H+2nbc, W+2nbc)
    """

    v4d = torch.nn.functional.pad(
        v4d, (nbc, nbc, nbc, nbc), mode="replicate"
    )                                                 # (1,1,H+2nbc,W+2nbc)
    return v4d



def adjust_sr(nx, dx, nbc, device):
    """
    构造 5 个炮点 + 70 个检波点的坐标，并转成网格索引 (int64)。
    返回: dict 包含 isx, isz, igx, igz (已含 nbc)
    """
    # 5 个炮点 (0,17,34,52,69 格点)；炮深 1 个网格（10m）
    sx_arr = torch.tensor([0, 17, 34, 52, 69], device=device) * dx
    sz_val = torch.tensor(1 * dx, device=device)

    gx_arr = torch.arange(nx, device=device) * dx
    gz_arr = torch.ones(nx, device=device) * dx

    round_int = lambda x: torch.floor(x / dx + 0.5).int()
    isx = round_int(sx_arr) + nbc                       # (5,)
    isz = round_int(sz_val)  + nbc                      # (   )
    igx = round_int(gx_arr) + nbc                       # (70,)
    igz = round_int(gz_arr) + nbc                       # (70,)

    return {"isx": isx, "isz": isz, "igx": igx, "igz": igz}



def a2d_mod_abc24_single_cuda(vpad: torch.Tensor,
                             wavelet: torch.Tensor, 
                             coord: dict,
                             params: dict,
                             *, 
                             device: str = "cuda") -> torch.Tensor:
    """
    单个模型的 CUDA 加速正演（对应 py_vel_to_seis.py 中的 a2d_mod_abc24_single）
    
    参数:
        vpad: (H_pad, W_pad) 填充后的速度模型  
        wavelet: (nt,) Ricker 子波
        coord: 包含 isx, isz, igx, igz 的坐标字典
        params: 参数字典，包含 dx, dt, nt, nbc, isFS
        device: 设备类型
    
    返回:
        torch.Tensor: (5, nt, 70) 地震数据
    """
    
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA 扩展不可用")
    

    isx = coord["isx"]
    isz = coord["isz"]
    igx = coord["igx"]
    igz = coord["igz"]


    # 调用 CUDA 扩展
    result = cuda_vel_forward_c.acoustic_forward_precomputed(
        vpad, wavelet, isx, isz, igx, igz,
        dx=params["dx"],
        dt=params["dt"], 
        nt=params["nt"],
        nbc=params["nbc"],
        isFS=params["isFS"]
    )
    
    return result


def a2d_mod_abc24_batch_cuda(vpad_batch: torch.Tensor,
                            wavelet: torch.Tensor, 
                            coord: dict,
                            params: dict,
                            *,
                            device: str = "cuda",
                            max_concurrent_batches: int = 4) -> torch.Tensor:
    """
    批量模型的 CUDA 加速正演，使用优化的batch kernel
    
    参数:
        vpad_batch: (B, H_pad, W_pad) 批量填充后的速度模型  
        wavelet: (nt,) Ricker 子波
        coord: 包含 isx, isz, igx, igz 的坐标字典
        params: 参数字典，包含 dx, dt, nt, nbc, isFS
        device: 设备类型
        max_concurrent_batches: 最大并发批次数量
    
    返回:
        torch.Tensor: (B, 5, nt, 70) 地震数据
    """
    
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA 扩展不可用")
    
    isx = coord["isx"]
    isz = coord["isz"]
    igx = coord["igx"]
    igz = coord["igz"]

    # 调用优化的批量 CUDA 扩展
    result = cuda_vel_forward_c.acoustic_forward_batch_optimized(
        vpad_batch, wavelet, isx, isz, igx, igz,
        dx=params["dx"],
        dt=params["dt"], 
        nt=params["nt"],
        nbc=params["nbc"],
        isFS=params["isFS"],
        max_concurrent_batches=max_concurrent_batches
    )
    
    return result

class Vel_Forward(nn.Module):
    def __init__(self, device="cuda", max_concurrent_batches=4, use_batch_kernel=True):
        super().__init__()
        # -------- 常量与缓存 --------
        self.params = {
            "dx": 10.0,
            "nbc": int(120),
            "nt": int(1000),
            "dt": 1e-3,
            "isFS": False,
        }
        self.wavelet = ricker(f=15.0, dt=self.params["dt"], nt=self.params["nt"],
                              device=device)

        self.coord  = adjust_sr(nx=70, dx=self.params["dx"], nbc=self.params["nbc"],
                                device=device)
        self.device = device
        self.max_concurrent_batches = max_concurrent_batches
        self.use_batch_kernel = use_batch_kernel


    @torch.no_grad()
    def forward(self, vel_b1hw):
        vel_b1hw = vel_b1hw.to(self.device)
        B = vel_b1hw.shape[0]
        vel_b1hw = padvel_bchw(vel_b1hw, self.params["nbc"])

        #t0 = time.time()

        if self.use_batch_kernel and CUDA_AVAILABLE:
            # 使用优化的batch kernel - 一次性处理所有批次
            vel_batch = vel_b1hw[:, 0]  # (B, H_pad, W_pad)

            result = a2d_mod_abc24_batch_cuda(
                vel_batch, self.wavelet, self.coord, self.params,
                device=self.device, max_concurrent_batches=self.max_concurrent_batches
            )  # (B, 5, nt, 70)

            #print(f"[INFO] batch forward done, B={B}, elapsed={time.time() - t0:.3f}s (CUDA batch kernel)")
            return result
        else:
            # 回退到逐个处理模式
            out_list = []
            for b in range(B):
                seis = a2d_mod_abc24_single_cuda(
                    vel_b1hw[b, 0], self.wavelet, self.coord, self.params, device=self.device
                )  # (5,nt,70)
                out_list.append(seis)

            #print(f"[INFO] sequential forward done, B={B}, elapsed={time.time() - t0:.3f}s (fallback mode)")
            return torch.stack(out_list)


def AbcCoef2D(vel, nbc, dx):
    """吸收边界系数 (PML 简化)"""
    if nbc <= 1:
        return torch.zeros_like(vel)

    nzbc, nxbc = vel.shape
    velmin     = vel.min()
    nz, nx     = nzbc - 2 * nbc, nxbc - 2 * nbc

    a     = (nbc - 1) * dx
    kappa = 3.0 * velmin * torch.log(torch.tensor(1e7, dtype=DTYPE, device=vel.device)) / (2.0 * a)

    damp1d = kappa * ((torch.arange(nbc, device=vel.device, dtype=DTYPE) * dx / a) ** 2)
    damp   = torch.zeros_like(vel)

    # 左右
    damp[:, :nbc]           = damp1d.flip(0).repeat(nzbc, 1)
    damp[:, nx + nbc:]      = damp1d.repeat(nzbc, 1)
    # 上下
    damp[:nbc, nbc:nx+nbc]  = damp1d.flip(0).unsqueeze(1).repeat(1, nx)
    damp[nz+nbc:, nbc:nx+nbc] = damp1d.unsqueeze(1).repeat(1, nx)
    return damp


# ------------------------------------------------------------
# 2. 单模型前向传播 (4-阶交错网格)
# ------------------------------------------------------------
@torch.no_grad()
def a2d_mod_abc24_single(vpad, wavelet, coord,  params, *, device):
    """
    v2d    : (70,70) float32 速度 (m/s)
    params : dict 统一管理常量
    return : (5, nt, 70) float32 => 5 炮、nt 时样、70 道
    """
    nbc, dx, dt, nt, isFS = params["nbc"], params["dx"], params["dt"], params["nt"], params["isFS"]


    abc    = AbcCoef2D(vpad, nbc, dx)
    alpha  = (vpad * dt / dx) ** 2
    kappa  = abc * dt
    beta_d = (vpad * dt) ** 2

    c1, c2, c3 = -2.5, 4.0 / 3.0, -1.0 / 12.0
    t1 = 2 + 2 * c1 * alpha - kappa
    t2 = 1 - kappa

    p0 = torch.zeros_like(vpad)
    p1 = torch.zeros_like(vpad)

    isx, isz, igx, igz = coord["isx"], coord["isz"], coord["igx"], coord["igz"]
    n_src   = isx.numel()
    n_rec   = igx.numel()
    seis    = torch.zeros((n_src, nt, n_rec), dtype=DTYPE, device=device)

    for src_idx in range(n_src):
        p0.zero_()
        p1.zero_()
        for it in range(nt):
            lap = (
                c2 * (
                    torch.roll(p1,  1, 1) + torch.roll(p1, -1, 1) +
                    torch.roll(p1,  1, 0) + torch.roll(p1, -1, 0)
                ) +
                c3 * (
                    torch.roll(p1,  2, 1) + torch.roll(p1, -2, 1) +
                    torch.roll(p1,  2, 0) + torch.roll(p1, -2, 0)
                )
            )
            p  = t1 * p1 - t2 * p0 + alpha * lap
            p[isz, isx[src_idx]] += beta_d[isz, isx[src_idx]] * wavelet[it]

            if isFS:                          # Free-surface
                p[nbc]     = 0.0
                p[nbc-1]   = -p[nbc+1]
                p[nbc-2]   = -p[nbc+2]

            seis[src_idx, it] = p[igz, igx]
            p0, p1 = p1, p

    return seis                               # (5, nt, 70)


# ------------------------------------------------------------
# 3. 批量接口 (B,1,70,70) -> (B,5,nt,70)
# ------------------------------------------------------------
def vel_to_seis_ref(vel_b1hw, *, device):
    """
    vel_b1hw : torch/np, shape (B,1,70,70)
    return    : np.ndarray (B,5,1000,70)
    """
    if isinstance(vel_b1hw, np.ndarray):
        vel_b1hw = torch.from_numpy(vel_b1hw)

    vel_b1hw = vel_b1hw.to(device, DTYPE)
    B        = vel_b1hw.shape[0]

    # -------- 常量与缓存 --------
    params = {
        "dx"      : 10.0,
        "nbc"     : 120,
        "nt"      : 1000,
        "dt"      : 1e-3,
        "isFS"    : False,
    }
    wavelet = ricker(f=15.0, dt=params["dt"], nt=params["nt"], device=device)
    coord   = adjust_sr(nx=70, dx=params["dx"], nbc=params["nbc"], device=device)
    vel_b1hw = padvel_bchw(vel_b1hw, params["nbc"])
    #print('vel_b1hw shape:', vel_b1hw.shape)
    # 下面的这个给实装成cuda 的, 最好是在B 维度上并行
    out_list = []
    t0 = time.time()
    for b in range(B):
        seis = a2d_mod_abc24_single(
            vel_b1hw[b, 0],wavelet,coord,  params, device=device
        )                                       # (5,nt,70)
        out_list.append(seis)

    print(f"[INFO] forward done, B={B}, elapsed={time.time()-t0:.3f}s")
    return torch.stack(out_list)

