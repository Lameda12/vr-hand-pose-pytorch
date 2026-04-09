/**
 * preprocess.cpp — High-performance BGR→normalized tensor preprocessing.
 *
 * Critical hot path: called per-frame at 30fps.
 * Target: <1ms @720p on i7-11th gen (vs ~3ms Python/NumPy).
 *
 * Operations:
 *   1. Resize to (W, H) via bilinear interpolation
 *   2. BGR → RGB channel swap (in-place SIMD-friendly loop)
 *   3. uint8 → float32, divide by 255
 *   4. ImageNet normalize: (x - mean) / std per channel
 *   5. HWC → CHW layout for PyTorch
 *
 * Exposed via pybind11 as:
 *   import preprocess_cpp
 *   tensor = preprocess_cpp.preprocess_bgr(frame_bgr_numpy, width, height)
 *   # Returns: float32 numpy [3, H, W], ready for torch.from_numpy()
 *
 * Build:
 *   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
 *
 * OpenCV CUDA path:
 *   When compiled with -DUSE_CUDA=ON and cv2.cuda is available,
 *   resize is offloaded to GPU (GpuMat pipeline).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

#ifdef USE_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// ImageNet normalization constants
static constexpr float MEAN_R = 0.485f;
static constexpr float MEAN_G = 0.456f;
static constexpr float MEAN_B = 0.406f;
static constexpr float STD_R  = 0.229f;
static constexpr float STD_G  = 0.224f;
static constexpr float STD_B  = 0.225f;
static constexpr float INV255 = 1.0f / 255.0f;


/**
 * Preprocess a BGR uint8 frame into a normalized CHW float32 array.
 *
 * @param frame_bgr  NumPy array [H_src, W_src, 3] uint8, BGR
 * @param out_w      Target width
 * @param out_h      Target height
 * @return           NumPy array [3, out_h, out_w] float32, RGB normalized
 *
 * Thread safety: stateless; safe to call from multiple Python threads
 *                (GIL released during heavy work).
 */
py::array_t<float> preprocess_bgr(
    py::array_t<uint8_t, py::array::c_contiguous> frame_bgr,
    int out_w,
    int out_h
) {
    // Validate input
    auto buf = frame_bgr.request();
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::invalid_argument(
            "Expected frame_bgr with shape [H, W, 3], got " +
            std::to_string(buf.ndim) + "D array"
        );
    }
    if (out_w <= 0 || out_h <= 0) {
        throw std::invalid_argument("Output dimensions must be positive");
    }

    const int src_h = static_cast<int>(buf.shape[0]);
    const int src_w = static_cast<int>(buf.shape[1]);

    // Wrap input in cv::Mat (zero-copy)
    cv::Mat src(src_h, src_w, CV_8UC3, buf.ptr);

    // Step 1: Resize (bilinear)
    cv::Mat resized;
    if (src_w == out_w && src_h == out_h) {
        resized = src;   // no-op: skip allocation
    } else {
        // Release GIL during resize (I/O intensive but not Python-safe)
        {
            py::gil_scoped_release release;
            cv::resize(src, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
        }
    }

    // Allocate output: [3, H, W] float32
    py::array_t<float> result({3, out_h, out_w});
    auto out_buf = result.mutable_unchecked<3>();   // bounds-checked in debug, fast in release

    // Step 2-5: BGR→RGB swap + normalize + CHW layout
    // Process row by row; compiler auto-vectorizes inner loop (AVX2 on modern CPUs).
    {
        py::gil_scoped_release release;

        const uint8_t* src_ptr = resized.data;
        const int row_stride   = resized.step[0];   // bytes per row (may have padding)

        // Channel offsets in output CHW layout
        float* ch_r = &out_buf(0, 0, 0);
        float* ch_g = &out_buf(1, 0, 0);
        float* ch_b = &out_buf(2, 0, 0);

        for (int y = 0; y < out_h; ++y) {
            const uint8_t* row = src_ptr + y * row_stride;
            for (int x = 0; x < out_w; ++x) {
                // OpenCV stores BGR; we output RGB
                const float b = row[x * 3 + 0] * INV255;
                const float g = row[x * 3 + 1] * INV255;
                const float r = row[x * 3 + 2] * INV255;

                const int idx = y * out_w + x;
                ch_r[idx] = (r - MEAN_R) / STD_R;
                ch_g[idx] = (g - MEAN_G) / STD_G;
                ch_b[idx] = (b - MEAN_B) / STD_B;
            }
        }
    }

    return result;
}


#ifdef USE_CUDA
/**
 * CUDA-accelerated preprocess path using cv::cuda::GpuMat.
 *
 * Offloads resize to GPU. Normalization runs on CPU (data transfer cost
 * only worth it when chaining with GPU inference).
 *
 * Only available when compiled with -DUSE_CUDA=ON.
 */
py::array_t<float> preprocess_bgr_cuda(
    py::array_t<uint8_t, py::array::c_contiguous> frame_bgr,
    int out_w,
    int out_h
) {
    auto buf = frame_bgr.request();
    const int src_h = static_cast<int>(buf.shape[0]);
    const int src_w = static_cast<int>(buf.shape[1]);

    cv::Mat src(src_h, src_w, CV_8UC3, buf.ptr);
    cv::cuda::GpuMat gpu_src, gpu_resized;

    {
        py::gil_scoped_release release;
        gpu_src.upload(src);
        cv::cuda::resize(gpu_src, gpu_resized, cv::Size(out_w, out_h),
                         0, 0, cv::INTER_LINEAR);
    }

    cv::Mat resized;
    {
        py::gil_scoped_release release;
        gpu_resized.download(resized);
    }

    // CPU normalization (same as CPU path)
    py::array_t<float> result({3, out_h, out_w});
    auto out_buf = result.mutable_unchecked<3>();

    {
        py::gil_scoped_release release;
        const uint8_t* src_ptr = resized.data;
        const int row_stride   = resized.step[0];
        float* ch_r = &out_buf(0, 0, 0);
        float* ch_g = &out_buf(1, 0, 0);
        float* ch_b = &out_buf(2, 0, 0);

        for (int y = 0; y < out_h; ++y) {
            const uint8_t* row = src_ptr + y * row_stride;
            for (int x = 0; x < out_w; ++x) {
                const float b = row[x * 3 + 0] * INV255;
                const float g = row[x * 3 + 1] * INV255;
                const float r = row[x * 3 + 2] * INV255;
                const int idx = y * out_w + x;
                ch_r[idx] = (r - MEAN_R) / STD_R;
                ch_g[idx] = (g - MEAN_G) / STD_G;
                ch_b[idx] = (b - MEAN_B) / STD_B;
            }
        }
    }

    return result;
}
#endif   // USE_CUDA


PYBIND11_MODULE(preprocess_cpp, m) {
    m.doc() = "Turin hand pose — C++ preprocessing hot path";

    m.def(
        "preprocess_bgr",
        &preprocess_bgr,
        py::arg("frame_bgr"),
        py::arg("out_w"),
        py::arg("out_h"),
        R"doc(
            Resize + BGR→RGB + ImageNet normalize → CHW float32.

            Args:
                frame_bgr: np.ndarray [H, W, 3] uint8 BGR
                out_w:     target width
                out_h:     target height

            Returns:
                np.ndarray [3, out_h, out_w] float32, RGB, ImageNet-normalized
                Ready for torch.from_numpy().unsqueeze(0)
        )doc"
    );

#ifdef USE_CUDA
    m.def(
        "preprocess_bgr_cuda",
        &preprocess_bgr_cuda,
        py::arg("frame_bgr"),
        py::arg("out_w"),
        py::arg("out_h"),
        "CUDA-accelerated variant (GPU resize + CPU normalize)."
    );
    m.attr("has_cuda") = true;
#else
    m.attr("has_cuda") = false;
#endif
}
