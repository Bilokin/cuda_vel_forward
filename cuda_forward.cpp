#include <torch/extension.h>
#include <vector>

namespace py = pybind11;


// 声明新的预计算版本 CUDA 函数
void cuda_acoustic_forward_precomputed(torch::Tensor vpad,
                                      torch::Tensor wavelet,
                                      torch::Tensor isx,
                                      torch::Tensor isz,
                                      torch::Tensor igx,
                                      torch::Tensor igz,
                                      torch::Tensor output,
                                      double dx, double dt, int nt, int nbc,
                                      bool isFS);

// 声明batch processing CUDA 函数
void cuda_acoustic_forward_batch(torch::Tensor vpad_batch,
                                torch::Tensor wavelet,
                                torch::Tensor isx,
                                torch::Tensor isz,
                                torch::Tensor igx,
                                torch::Tensor igz,
                                torch::Tensor output_batch,
                                double dx, double dt, int nt, int nbc,
                                bool isFS);

void cuda_acoustic_forward_batch_optimized(torch::Tensor vpad_batch,
                                          torch::Tensor wavelet,
                                          torch::Tensor isx,
                                          torch::Tensor isz,
                                          torch::Tensor igx,
                                          torch::Tensor igz,
                                          torch::Tensor output_batch,
                                          double dx, double dt, int nt, int nbc,
                                          bool isFS, int max_concurrent_batches);


// 新的预计算版本函数
torch::Tensor acoustic_forward_precomputed_cuda(torch::Tensor vpad,
                                               torch::Tensor wavelet,
                                               torch::Tensor isx,
                                               torch::Tensor isz,
                                               torch::Tensor igx,
                                               torch::Tensor igz,
                                               double dx = 10.0f,
                                               double dt = 1e-3f,
                                               int nt = 1000,
                                               int nbc = 120,
                                               bool isFS = false) {
    
    // 检查输入
    TORCH_CHECK(vpad.dim() == 2, "vpad must be 2D");
    TORCH_CHECK(wavelet.dim() == 1, "wavelet must be 1D");
    TORCH_CHECK(isx.dim() == 1, "isx must be 1D");
    TORCH_CHECK(isz.dim() == 0, "isz must be scalar");
    TORCH_CHECK(igx.dim() == 1, "igx must be 1D");
    TORCH_CHECK(igz.dim() == 1, "igz must be 1D");
    TORCH_CHECK(vpad.device().is_cuda(), "vpad must be on CUDA device");
    
    int n_src = isx.size(0);
    int n_rec = igx.size(0);
    
    // 创建输出张量 (n_src, nt, n_rec)
    auto options = torch::TensorOptions()
                      .dtype(torch::kFloat64)
                      .device(vpad.device());
    auto output = torch::zeros({n_src, nt, n_rec}, options);
    
    // 调用 CUDA kernel
    cuda_acoustic_forward_precomputed(vpad, wavelet, isx, isz, igx, igz, 
                                     output, dx, dt, nt, nbc, isFS);
    
    return output;
}

// Batch processing 函数
torch::Tensor acoustic_forward_batch_cuda(torch::Tensor vpad_batch,
                                         torch::Tensor wavelet,
                                         torch::Tensor isx,
                                         torch::Tensor isz,
                                         torch::Tensor igx,
                                         torch::Tensor igz,
                                         double dx = 10.0f,
                                         double dt = 1e-3f,
                                         int nt = 1000,
                                         int nbc = 120,
                                         bool isFS = false) {
    
    // 检查输入
    TORCH_CHECK(vpad_batch.dim() == 3, "vpad_batch must be 3D (B, H, W)");
    TORCH_CHECK(wavelet.dim() == 1, "wavelet must be 1D");
    TORCH_CHECK(isx.dim() == 1, "isx must be 1D");
    TORCH_CHECK(isz.dim() == 0, "isz must be scalar");
    TORCH_CHECK(igx.dim() == 1, "igx must be 1D");
    TORCH_CHECK(igz.dim() == 1, "igz must be 1D");
    TORCH_CHECK(vpad_batch.device().is_cuda(), "vpad_batch must be on CUDA device");
    
    int B = vpad_batch.size(0);
    int n_src = isx.size(0);
    int n_rec = igx.size(0);
    
    // 创建输出张量 (B, n_src, nt, n_rec)
    auto options = torch::TensorOptions()
                      .dtype(torch::kFloat64)
                      .device(vpad_batch.device());
    auto output_batch = torch::zeros({B, n_src, nt, n_rec}, options);
    
    // 调用 CUDA batch kernel
    cuda_acoustic_forward_batch(vpad_batch, wavelet, isx, isz, igx, igz, 
                               output_batch, dx, dt, nt, nbc, isFS);
    
    return output_batch;
}

// 优化版本的batch processing函数
torch::Tensor acoustic_forward_batch_optimized_cuda(torch::Tensor vpad_batch,
                                                   torch::Tensor wavelet,
                                                   torch::Tensor isx,
                                                   torch::Tensor isz,
                                                   torch::Tensor igx,
                                                   torch::Tensor igz,
                                                   double dx = 10.0f,
                                                   double dt = 1e-3f,
                                                   int nt = 1000,
                                                   int nbc = 120,
                                                   bool isFS = false,
                                                   int max_concurrent_batches = 4) {
    
    // 检查输入
    TORCH_CHECK(vpad_batch.dim() == 3, "vpad_batch must be 3D (B, H, W)");
    TORCH_CHECK(wavelet.dim() == 1, "wavelet must be 1D");
    TORCH_CHECK(isx.dim() == 1, "isx must be 1D");
    TORCH_CHECK(isz.dim() == 0, "isz must be scalar");
    TORCH_CHECK(igx.dim() == 1, "igx must be 1D");
    TORCH_CHECK(igz.dim() == 1, "igz must be 1D");
    TORCH_CHECK(vpad_batch.device().is_cuda(), "vpad_batch must be on CUDA device");
    
    int B = vpad_batch.size(0);
    int n_src = isx.size(0);
    int n_rec = igx.size(0);
    
    // 创建输出张量 (B, n_src, nt, n_rec)
    auto options = torch::TensorOptions()
                      .dtype(torch::kFloat64)
                      .device(vpad_batch.device());
    auto output_batch = torch::zeros({B, n_src, nt, n_rec}, options);
    
    // 调用优化的 CUDA batch kernel
    cuda_acoustic_forward_batch_optimized(vpad_batch, wavelet, isx, isz, igx, igz, 
                                         output_batch, dx, dt, nt, nbc, isFS, 
                                         max_concurrent_batches);
    
    return output_batch;
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("acoustic_forward_precomputed", &acoustic_forward_precomputed_cuda,
          "Acoustic forward modeling with precomputed parameters (CUDA)",
          py::arg("vpad"),
          py::arg("wavelet"),
          py::arg("isx"),
          py::arg("isz"),
          py::arg("igx"),
          py::arg("igz"),
          py::arg("dx") = 10.0f,
          py::arg("dt") = 1e-3f,
          py::arg("nt") = 1000,
          py::arg("nbc") = 120,
          py::arg("isFS") = false);

    m.def("acoustic_forward_batch", &acoustic_forward_batch_cuda,
          "Batch acoustic forward modeling with CUDA streams (CUDA)",
          py::arg("vpad_batch"),
          py::arg("wavelet"),
          py::arg("isx"),
          py::arg("isz"),
          py::arg("igx"),
          py::arg("igz"),
          py::arg("dx") = 10.0f,
          py::arg("dt") = 1e-3f,
          py::arg("nt") = 1000,
          py::arg("nbc") = 120,
          py::arg("isFS") = false);

    m.def("acoustic_forward_batch_optimized", &acoustic_forward_batch_optimized_cuda,
          "Optimized batch acoustic forward modeling with advanced CUDA streams (CUDA)",
          py::arg("vpad_batch"),
          py::arg("wavelet"),
          py::arg("isx"),
          py::arg("isz"),
          py::arg("igx"),
          py::arg("igz"),
          py::arg("dx") = 10.0f,
          py::arg("dt") = 1e-3f,
          py::arg("nt") = 1000,
          py::arg("nbc") = 120,
          py::arg("isFS") = false,
          py::arg("max_concurrent_batches") = 4);
} 
