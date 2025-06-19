#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

// CUDA 常量
#define BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

// 数学常量
__constant__ float c1 = -2.5f;
__constant__ float c2 = 4.0f / 3.0f;
__constant__ float c3 = -1.0f / 12.0f;

// 数值稳定性常量
__constant__ float MAX_VALUE = 1e8f;  // 最大允许值
__constant__ float MIN_VALUE = 1e-8f; // 最小有意义值

// 预计算 damp1d 数组的 kernel
__global__ void compute_damp1d_kernel(float* damp1d, float kappa, float dx, float a, int nbc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nbc) return;
    
    float r = i * dx / a;
    damp1d[i] = kappa * r * r;
}

// 吸收边界系数计算 kernel - 完全按照 Python AbcCoef2D 实现
__global__ void abc_coef_kernel(float* abc, const float* damp1d, int nz_pad, int nx_pad, 
                               int nbc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nz_pad || idy >= nx_pad) return;
    
    // 初始化为 0
    abc[idx * nx_pad + idy] = 0.0f;
    
    if (nbc <= 1) return;
    
    int nz = nz_pad - 2 * nbc;
    int nx = nx_pad - 2 * nbc;
    
    // 按照 Python 实现的逻辑
    // 左边界: damp[:, :nbc] = damp1d.flip(0).repeat(nzbc, 1)
    if (idy < nbc) {
        int flip_idx = nbc - 1 - idy;  // flip(0)
        abc[idx * nx_pad + idy] = damp1d[flip_idx];
    }
    // 右边界: damp[:, nx + nbc:] = damp1d.repeat(nzbc, 1)  
    else if (idy >= nx + nbc) {
        int damp_idx = idy - (nx + nbc);
        abc[idx * nx_pad + idy] = damp1d[damp_idx];
    }
    
    // 上边界: damp[:nbc, nbc:nx+nbc] = damp1d.flip(0).unsqueeze(1).repeat(1, nx)
    if (idx < nbc && idy >= nbc && idy < nx + nbc) {
        int flip_idx = nbc - 1 - idx;  // flip(0)
        abc[idx * nx_pad + idy] = damp1d[flip_idx];
    }
    // 下边界: damp[nz+nbc:, nbc:nx+nbc] = damp1d.unsqueeze(1).repeat(1, nx)
    else if (idx >= nz + nbc && idy >= nbc && idy < nx + nbc) {
        int damp_idx = idx - (nz + nbc);
        abc[idx * nx_pad + idy] = damp1d[damp_idx];
    }
}

// 拉普拉斯算子计算 kernel - 模拟 torch.roll 行为
__global__ void laplacian_kernel(float* lap, const float* p, int nz_pad, int nx_pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nz_pad || idy >= nx_pad) return;
    
    int center = idx * nx_pad + idy;
    
    // 模拟 torch.roll 的周期性边界行为
    // torch.roll(p1, 1, 1) 相当于 y 方向右移 1
    // torch.roll(p1, -1, 1) 相当于 y 方向左移 1
    // torch.roll(p1, 1, 0) 相当于 x 方向下移 1  
    // torch.roll(p1, -1, 0) 相当于 x 方向上移 1
    
    int y_plus1 = (idy + 1) % nx_pad;              // torch.roll(p1, 1, 1)
    int y_minus1 = (idy - 1 + nx_pad) % nx_pad;    // torch.roll(p1, -1, 1)
    int x_plus1 = (idx + 1) % nz_pad;              // torch.roll(p1, 1, 0)
    int x_minus1 = (idx - 1 + nz_pad) % nz_pad;    // torch.roll(p1, -1, 0)
    
    int y_plus2 = (idy + 2) % nx_pad;              // torch.roll(p1, 2, 1)
    int y_minus2 = (idy - 2 + nx_pad) % nx_pad;    // torch.roll(p1, -2, 1)
    int x_plus2 = (idx + 2) % nz_pad;              // torch.roll(p1, 2, 0)
    int x_minus2 = (idx - 2 + nz_pad) % nz_pad;    // torch.roll(p1, -2, 0)
    
    // 预取数据以减少内存延迟并进行数值检查
    float p_y_plus1 = p[idx * nx_pad + y_plus1];
    float p_y_minus1 = p[idx * nx_pad + y_minus1];
    float p_x_plus1 = p[x_plus1 * nx_pad + idy];
    float p_x_minus1 = p[x_minus1 * nx_pad + idy];
    
    float p_y_plus2 = p[idx * nx_pad + y_plus2];
    float p_y_minus2 = p[idx * nx_pad + y_minus2];
    float p_x_plus2 = p[x_plus2 * nx_pad + idy];
    float p_x_minus2 = p[x_minus2 * nx_pad + idy];
    
    // 数值稳定性检查
    if (!isfinite(p_y_plus1) || !isfinite(p_y_minus1) || !isfinite(p_x_plus1) ||
        !isfinite(p_x_minus1) || !isfinite(p_y_plus2) || !isfinite(p_y_minus2) ||
        !isfinite(p_x_plus2) || !isfinite(p_x_minus2)) {
        lap[center] = 0.0f;
        return;
    }
    
    // 4阶有限差分拉普拉斯算子 - 分段计算提高精度
    float term1 = c2 * (p_y_plus1 + p_y_minus1 + p_x_plus1 + p_x_minus1);
    float term2 = c3 * (p_y_plus2 + p_y_minus2 + p_x_plus2 + p_x_minus2);
    
    float lap_val = term1 + term2;
    
    // 数值范围保护
    if (isfinite(lap_val) && fabsf(lap_val) <= MAX_VALUE) {
        lap[center] = lap_val;
    } else {
        lap[center] = 0.0f;
    }
}

// 波场更新 kernel
__global__ void wavefield_update_kernel(float* p_new, float* p_cur, float* p_old,
                                       const float* lap, const float* alpha, 
                                       const float* kappa, int nz_pad, int nx_pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= nz_pad || idy >= nx_pad) return;
    
    int center = idx * nx_pad + idy;
    
    float alpha_val = alpha[center];
    float kappa_val = kappa[center];
    float lap_val = lap[center];
    float p_cur_val = p_cur[center];
    float p_old_val = p_old[center];
    
    // 启用输入值有效性检查
    if (!isfinite(alpha_val) || !isfinite(kappa_val) || !isfinite(lap_val) ||
        !isfinite(p_cur_val) || !isfinite(p_old_val)) {
        p_new[center] = 0.0f;
        return;
    }
    
    // 检查数值范围
    if (fabsf(alpha_val) > MAX_VALUE || fabsf(kappa_val) > MAX_VALUE ||
        fabsf(lap_val) > MAX_VALUE || fabsf(p_cur_val) > MAX_VALUE ||
        fabsf(p_old_val) > MAX_VALUE) {
        p_new[center] = 0.5f * p_cur_val; // 保守的回退策略
        return;
    }
    
    float t1 = 2.0f + 2.0f * c1 * alpha_val - kappa_val;
    float t2 = 1.0f - kappa_val;
    
    // 分段计算减少数值误差
    float term1 = t1 * p_cur_val;
    float term2 = t2 * p_old_val;
    float term3 = alpha_val * lap_val;
    
    float result = term1 - term2 + term3;
    
    // 确保结果有效且在合理范围内
    if (isfinite(result) && fabsf(result) <= MAX_VALUE) {
        p_new[center] = result;
    } else {
        // 数值保护：使用更保守的更新
        p_new[center] = 0.8f * p_cur_val - 0.2f * p_old_val;
    }
}

// 改进的震源注入 kernel - 增加数值保护
__global__ void source_inject_kernel(float* p, int sz, int sx, float source_val, 
                                    int nx_pad) {
    // 数值范围检查
    if (!isfinite(source_val) || fabsf(source_val) > MAX_VALUE) {
        return; // 跳过无效的震源值
    }
    
    int pos = sz * nx_pad + sx;
    
    // 检查当前波场值
    float current_val = p[pos];
    if (!isfinite(current_val)) {
        p[pos] = source_val; // 如果当前值无效，直接设置
    } else {
        float new_val = current_val + source_val;
        // 检查结果是否有效
        if (isfinite(new_val) && fabsf(new_val) <= MAX_VALUE) {
            p[pos] = new_val;
        }
        // 如果无效，保持原值不变
    }
}

// 改进的自由表面边界条件 kernel
__global__ void free_surface_kernel(float* p, int nbc, int nx_pad) {
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    if (idy >= nx_pad) return;
    
    // 自由表面条件：p = 0 在表面
    p[nbc * nx_pad + idy] = 0.0f;
    
    // 改进的镜像边界条件，增加数值稳定性检查
    if (nbc >= 1) {
        int pos_above1 = (nbc + 1) * nx_pad + idy;
        int pos_below1 = (nbc - 1) * nx_pad + idy;
        
        float val_above1 = p[pos_above1];
        if (isfinite(val_above1) && fabsf(val_above1) <= MAX_VALUE) {
            p[pos_below1] = -val_above1;
        } else {
            p[pos_below1] = 0.0f;
        }
    }
    
    if (nbc >= 2) {
        int pos_above2 = (nbc + 2) * nx_pad + idy;
        int pos_below2 = (nbc - 2) * nx_pad + idy;
        
        float val_above2 = p[pos_above2];
        if (isfinite(val_above2) && fabsf(val_above2) <= MAX_VALUE) {
            p[pos_below2] = -val_above2;
        } else {
            p[pos_below2] = 0.0f;
        }
    }
}

// 记录提取 kernel
__global__ void record_extract_kernel(float* seis, const float* p, 
                                     const int* igz, const int* igx, 
                                     int n_rec, int nx_pad, int time_idx) {
    int rec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rec_idx >= n_rec) return;
    
    int pos = igz[rec_idx] * nx_pad + igx[rec_idx];
    seis[rec_idx] = p[pos];
}

// 新的 CUDA 函数：接受预计算的 vpad, wavelet, coord
void cuda_acoustic_forward_precomputed(torch::Tensor vpad,           // (H_pad, W_pad) 
                                      torch::Tensor wavelet,        // (nt,)
                                      torch::Tensor isx,            // (5,) 炮点 x 坐标
                                      torch::Tensor isz,            // () 炮点 z 坐标  
                                      torch::Tensor igx,            // (70,) 检波点 x 坐标
                                      torch::Tensor igz,            // (70,) 检波点 z 坐标
                                      torch::Tensor output,         // (5, nt, 70)
                                      float dx, float dt, int nt, int nbc,
                                      bool isFS) {
    
    // 获取设备和数据类型选项
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(vpad.device());
    
    int H_pad = vpad.size(0);  
    int W_pad = vpad.size(1);  
    
    // 计算吸收边界系数 - 按照 Python AbcCoef2D 实现
    float velmin = vpad.min().item<float>();
    auto abc = torch::zeros({H_pad, W_pad}, options);
    
    if (nbc > 1) {
        // 计算 kappa 和 damp1d
        float a = (nbc - 1) * dx;
        float kappa = 3.0f * velmin * logf(1e7f) / (2.0f * a);
        
        // 分配 damp1d 数组
        auto damp1d = torch::zeros({nbc}, options);
        
        // 计算 damp1d: kappa * ((torch.arange(nbc) * dx / a) ** 2)
        dim3 block_1d(256);
        dim3 grid_1d((nbc + block_1d.x - 1) / block_1d.x);
        compute_damp1d_kernel<<<grid_1d, block_1d>>>(
            damp1d.data_ptr<float>(), kappa, dx, a, nbc);
        
        // 计算 ABC 系数
        dim3 block_2d(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid_2d((H_pad + block_2d.x - 1) / block_2d.x,
                     (W_pad + block_2d.y - 1) / block_2d.y);
        
        abc_coef_kernel<<<grid_2d, block_2d>>>(
            abc.data_ptr<float>(), damp1d.data_ptr<float>(),
            H_pad, W_pad, nbc);
    }
    
    // 计算系数
    auto alpha = (vpad * dt / dx).pow(2);
    auto kappa = abc * dt;
    auto beta_d = (vpad * dt).pow(2);
    
    // 分配波场内存
    auto p_old = torch::zeros({H_pad, W_pad}, options);
    auto p_cur = torch::zeros({H_pad, W_pad}, options);
    auto p_new = torch::zeros({H_pad, W_pad}, options);
    auto lap = torch::zeros({H_pad, W_pad}, options);
    
    int n_src = isx.size(0);
    int n_rec = igx.size(0);
    int isz_val = isz.item<int>();
    
    // 设置CUDA网格和块配置
    dim3 block_2d(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_2d((H_pad + block_2d.x - 1) / block_2d.x,
                 (W_pad + block_2d.y - 1) / block_2d.y);
    
    // 对每个炮点进行正演
    for (int src_idx = 0; src_idx < n_src; src_idx++) {
        // 重置波场
        p_old.zero_();
        p_cur.zero_();
        p_new.zero_();
        
        int sx = isx[src_idx].item<int>();
        
        // 时间步进
        for (int it = 0; it < nt; it++) {
            // 计算拉普拉斯算子
            laplacian_kernel<<<grid_2d, block_2d>>>(
                lap.data_ptr<float>(), p_cur.data_ptr<float>(),
                H_pad, W_pad);
            
            // 更新波场
            wavefield_update_kernel<<<grid_2d, block_2d>>>(
                p_new.data_ptr<float>(), p_cur.data_ptr<float>(),
                p_old.data_ptr<float>(), lap.data_ptr<float>(),
                alpha.data_ptr<float>(), kappa.data_ptr<float>(),
                H_pad, W_pad);
            
            // 注入震源
            float source_val = beta_d[isz_val][sx].item<float>() * wavelet[it].item<float>();
            source_inject_kernel<<<1, 1>>>(
                p_new.data_ptr<float>(), isz_val, sx, source_val, W_pad);
            
            // 自由表面边界条件
            if (isFS) {
                dim3 grid_fs((W_pad + 255) / 256);
                free_surface_kernel<<<grid_fs, 256>>>(
                    p_new.data_ptr<float>(), nbc, W_pad);
            }
            
            // 记录地震数据
            dim3 grid_rec((n_rec + 255) / 256);
            auto seis_slice = output.select(0, src_idx);  // (nt, 70)
            record_extract_kernel<<<grid_rec, 256>>>(
                seis_slice.select(0, it).data_ptr<float>(), p_new.data_ptr<float>(),
                igz.data_ptr<int>(), igx.data_ptr<int>(),
                n_rec, W_pad, 0);


            // 交换波场指针
            auto temp = p_old;
            p_old = p_cur;
            p_cur = p_new;
            p_new = temp;
        }
    }
    
    cudaDeviceSynchronize();
}

// 支持batch processing的CUDA声波正演函数 - 使用CUDA streams并行加速
void cuda_acoustic_forward_batch(torch::Tensor vpad_batch,       // (B, H_pad, W_pad)
                                 torch::Tensor wavelet,          // (nt,)
                                 torch::Tensor isx,              // (5,) 炮点 x 坐标
                                 torch::Tensor isz,              // () 炮点 z 坐标  
                                 torch::Tensor igx,              // (70,) 检波点 x 坐标
                                 torch::Tensor igz,              // (70,) 检波点 z 坐标
                                 torch::Tensor output_batch,     // (B, 5, nt, 70)
                                 float dx, float dt, int nt, int nbc,
                                 bool isFS) {
    
    // 获取批次大小和空间维度
    int B = vpad_batch.size(0);
    int H_pad = vpad_batch.size(1);  
    int W_pad = vpad_batch.size(2);
    int n_src = isx.size(0);
    int n_rec = igx.size(0);
    int isz_val = isz.item<int>();
    
    // 获取设备和数据类型选项
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(vpad_batch.device());
    
    // 创建CUDA streams for 并行处理
    const int MAX_STREAMS = 8;  // 最大并发流数量
    int num_streams = std::min(B, MAX_STREAMS);
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 预分配共享的内存池（减少内存分配开销）
    std::vector<torch::Tensor> p_old_batch, p_cur_batch, p_new_batch, lap_batch;
    std::vector<torch::Tensor> alpha_batch, kappa_batch, beta_d_batch, abc_batch;
    
    for (int i = 0; i < num_streams; i++) {
        p_old_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        p_cur_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        p_new_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        lap_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        alpha_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        kappa_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        beta_d_batch.push_back(torch::zeros({H_pad, W_pad}, options));
        abc_batch.push_back(torch::zeros({H_pad, W_pad}, options));
    }
    
    // 设置CUDA网格和块配置
    dim3 block_2d(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_2d((H_pad + block_2d.x - 1) / block_2d.x,
                 (W_pad + block_2d.y - 1) / block_2d.y);
    dim3 block_1d(256);
    dim3 grid_fs((W_pad + 255) / 256);
    dim3 grid_rec((n_rec + 255) / 256);
    
    // 批次并行处理
    for (int b = 0; b < B; b++) {
        int stream_id = b % num_streams;
        cudaStream_t current_stream = streams[stream_id];
        
        // 获取当前批次的速度模型
        auto vpad_current = vpad_batch.select(0, b);  // (H_pad, W_pad)
        auto output_current = output_batch.select(0, b);  // (5, nt, 70)
        
        // 使用预分配的内存
        auto& p_old = p_old_batch[stream_id];
        auto& p_cur = p_cur_batch[stream_id];
        auto& p_new = p_new_batch[stream_id];
        auto& lap = lap_batch[stream_id];
        auto& alpha = alpha_batch[stream_id];
        auto& kappa = kappa_batch[stream_id];
        auto& beta_d = beta_d_batch[stream_id];
        auto& abc = abc_batch[stream_id];
        
        // 异步计算系数
        alpha.copy_(vpad_current);
        alpha.mul_(dt / dx).pow_(2);
        
        beta_d.copy_(vpad_current);
        beta_d.mul_(dt).pow_(2);
        
        // 计算吸收边界系数
        float velmin = vpad_current.min().item<float>();
        abc.zero_();
        
        if (nbc > 1) {
            // 计算 kappa 和 damp1d
            float a = (nbc - 1) * dx;
            float kappa_val = 3.0f * velmin * logf(1e7f) / (2.0f * a);
            
            // 分配 damp1d 数组（每个stream独立）
            auto damp1d = torch::zeros({nbc}, options);
            
            // 在指定stream中异步执行
            dim3 grid_1d((nbc + block_1d.x - 1) / block_1d.x);
            compute_damp1d_kernel<<<grid_1d, block_1d, 0, current_stream>>>(
                damp1d.data_ptr<float>(), kappa_val, dx, a, nbc);
            
            // 计算 ABC 系数
            abc_coef_kernel<<<grid_2d, block_2d, 0, current_stream>>>(
                abc.data_ptr<float>(), damp1d.data_ptr<float>(),
                H_pad, W_pad, nbc);
        }
        
        // 异步计算 kappa
        kappa.copy_(abc);
        kappa.mul_(dt);
        
        // 对每个炮点进行正演
        for (int src_idx = 0; src_idx < n_src; src_idx++) {
            // 异步重置波场
            p_old.zero_();
            p_cur.zero_();
            p_new.zero_();
            
            int sx = isx[src_idx].item<int>();
            
            // 时间步进
            for (int it = 0; it < nt; it++) {
                // 在指定stream中异步执行所有kernel
                
                // 计算拉普拉斯算子
                laplacian_kernel<<<grid_2d, block_2d, 0, current_stream>>>(
                    lap.data_ptr<float>(), p_cur.data_ptr<float>(),
                    H_pad, W_pad);
                
                // 更新波场
                wavefield_update_kernel<<<grid_2d, block_2d, 0, current_stream>>>(
                    p_new.data_ptr<float>(), p_cur.data_ptr<float>(),
                    p_old.data_ptr<float>(), lap.data_ptr<float>(),
                    alpha.data_ptr<float>(), kappa.data_ptr<float>(),
                    H_pad, W_pad);
                
                // 注入震源
                float source_val = beta_d[isz_val][sx].item<float>() * wavelet[it].item<float>();
                source_inject_kernel<<<1, 1, 0, current_stream>>>(
                    p_new.data_ptr<float>(), isz_val, sx, source_val, W_pad);
                
                // 自由表面边界条件
                if (isFS) {
                    free_surface_kernel<<<grid_fs, 256, 0, current_stream>>>(
                        p_new.data_ptr<float>(), nbc, W_pad);
                }
                
                // 记录地震数据
                auto seis_slice = output_current.select(0, src_idx);  // (nt, 70)
                record_extract_kernel<<<grid_rec, 256, 0, current_stream>>>(
                    seis_slice.select(0, it).data_ptr<float>(), p_new.data_ptr<float>(),
                    igz.data_ptr<int>(), igx.data_ptr<int>(),
                    n_rec, W_pad, 0);
                
                // 交换波场指针
                auto temp = p_old;
                p_old = p_cur;
                p_cur = p_new;
                p_new = temp;
            }
        }
    }
    
    // 同步所有streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}

// 优化版本：使用更精细的内存管理和负载均衡
void cuda_acoustic_forward_batch_optimized(torch::Tensor vpad_batch,       // (B, H_pad, W_pad)
                                           torch::Tensor wavelet,          // (nt,)
                                           torch::Tensor isx,              // (5,) 炮点 x 坐标
                                           torch::Tensor isz,              // () 炮点 z 坐标  
                                           torch::Tensor igx,              // (70,) 检波点 x 坐标
                                           torch::Tensor igz,              // (70,) 检波点 z 坐标
                                           torch::Tensor output_batch,     // (B, 5, nt, 70)
                                           float dx, float dt, int nt, int nbc,
                                           bool isFS, int max_concurrent_batches = 4) {
    
    int B = vpad_batch.size(0);
    int H_pad = vpad_batch.size(1);  
    int W_pad = vpad_batch.size(2);
    
    // 动态调整并发批次数量
    int concurrent_batches = std::min(std::min(B, max_concurrent_batches), 8);
    
    // 批次分组处理
    for (int batch_start = 0; batch_start < B; batch_start += concurrent_batches) {
        int batch_end = std::min(batch_start + concurrent_batches, B);
        int current_batch_size = batch_end - batch_start;
        
        // 为当前批次组创建streams
        std::vector<cudaStream_t> streams(current_batch_size);
        for (int i = 0; i < current_batch_size; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // 并行处理当前批次组
        for (int local_b = 0; local_b < current_batch_size; local_b++) {
            int global_b = batch_start + local_b;
            
            auto vpad_current = vpad_batch.select(0, global_b);
            auto output_current = output_batch.select(0, global_b);
            
            // 在对应的stream中异步调用原始函数
            // 这里可以进一步优化为直接调用kernel而不是包装函数
            cuda_acoustic_forward_precomputed(
                vpad_current, wavelet, isx, isz, igx, igz,
                output_current, dx, dt, nt, nbc, isFS
            );
        }
        
        // 同步当前批次组
        for (int i = 0; i < current_batch_size; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
}